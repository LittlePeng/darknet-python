#include <Python.h>
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "blas.h"

static char **pyg_names = NULL;
static image **pyg_alphabet = NULL;
static network pyg_net;


static void pyload_network(char *datacfg, char *cfgfile, char *weightfile)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    pyg_names = get_labels(name_list);
    
    pyg_alphabet = load_alphabet();
    pyg_net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&pyg_net, weightfile);
    }
    set_batch_network(&pyg_net, 1);
}

static void pydetect_file(char* filename, char* outfile, float thresh) {
    clock_t time;
    int j;
    float nms=.4;
    float hier_thresh = .4;
    
    image im = load_image_color(filename,0,0);
    image sized = letterbox_image(im, pyg_net.w, pyg_net.h);

    
    layer l = pyg_net.layers[pyg_net.n-1];
    
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));
    
    float *X = sized.data;
    time=clock();
    network_predict(pyg_net, X);
    printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh, 0);
    if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    //else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs,
                    pyg_names, pyg_alphabet, l.classes);
    if(outfile){
        save_image(im, outfile);
    }
    else{
        save_image(im, "predictions");
    }
    
    free_image(im);
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);
}


static PyObject * pydetect(image im, char* outfile, float thresh) {
    clock_t time;
    int j;
    float nms=.4;
    float hier_thresh = .4;
    
    // 416*416
    image sized = letterbox_image(im, pyg_net.w, pyg_net.h);
    
    layer l = pyg_net.layers[pyg_net.n-1];
    
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));
    
    float *X = sized.data;
    time=clock();
    network_predict(pyg_net, X);
    printf("%s: Predicted in %f seconds.\n", "image", sec(clock()-time));
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh, 0);
    
    if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    
    //draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs,
    //                pyg_names, pyg_alphabet, l.classes);
    
    int num = l.w*l.h*l.n;

    int lsize = 0;
    for(int i = 0; i < num; ++i){
        int class = max_index(probs[i], l.classes);
        float prob = probs[i][class];
        if(prob > thresh){
            lsize ++;
        }
    }
    
    assert(pyg_net.w == pyg_net.h);
    float rate = (float) (im.w > im.h ? im.w : im.h) / pyg_net.w;
    int pending_tb = 0;
    int pending_lr = 0;
    if(im.w > im.h){
        pending_tb = (float)(im.w - im.h)/(2*im.w) * pyg_net.w;
    }else{
        pending_lr = (float)(im.h - im.w)/(2*im.h) * pyg_net.w;
    }
    //printf("net: %d %d, rate:%f \n", pyg_net.w, pyg_net.h, rate);
    PyObject *retval = PyList_New(lsize); // The returned Python object
    int lindex = 0;
    for(int i = 0; i < num; ++i){
        int class = max_index(probs[i], l.classes);
        float prob = probs[i][class];
        if(prob > thresh){
            box b = boxes[i];
            
            int left  = ((b.x-b.w/2.)*sized.w - pending_lr) * rate;
            int right = ((b.x+b.w/2.)*sized.w - pending_lr) * rate;
            int top   = ((b.y-b.h/2.)*sized.h - pending_tb) * rate;
            int bot   = ((b.y+b.h/2.)*sized.h - pending_tb) * rate;
            
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;
            
            PyList_SetItem(retval, lindex++, Py_BuildValue("sfiiii", pyg_names[class], prob, left, right, top, bot)); // Add results to list
        }
    }

    free_image(im);
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);
    return retval;
}

static PyObject *
libpydarknet_load(PyObject *self, PyObject *args)
{
    char *cfg_file;
    char *model_file;
    int sts = 0;
    
    if (!PyArg_ParseTuple(args, "ss", &cfg_file, &model_file)){
        printf("bad args\n");
        return NULL;
    }
    pyload_network("cfg/coco.data", cfg_file, model_file);
    return Py_BuildValue("i", sts);
}

static PyObject *
libpydarknet_detect_file(PyObject *self, PyObject *args)
{
    char* file_path;
    if(pyg_names == NULL){
        printf("weights not load!\n");
        return NULL;
    }
    if (!PyArg_ParseTuple(args, "s", &file_path)){
        printf("bad args\n");
        return NULL;
    }
    pydetect_file(file_path, NULL, 0.2);
    return Py_BuildValue("i", 1);
}

static PyObject *
libpydarknet_detect(PyObject *self, PyObject *args)
{
    if(pyg_names == NULL){
        printf("weights not load!\n");
        return NULL;
    }
    PyObject *o;
    int width;
    int height;
    int channel;
    float threshold;
    if (!PyArg_ParseTuple(args, "Oiiif", &o, &width, &height, & channel, &threshold)){
        printf("bad args\n");
        return NULL;
    }
    Py_buffer buffer;
    if(PyObject_GetBuffer(o, &buffer, PyBUF_CONTIG)){
        printf("bad buffer args\n");
        return NULL;
    }
    int size = height*width*channel;
    //printf("width = %d height =%d channel=%d threshold=%f size=%zd\n", width, height, channel, threshold, size);
    
    
    image img = make_image(width, height, channel);
    int count = 0;
    int step = width*channel;
    
    unsigned char* data = (unsigned char*)buffer.buf;
    for(int k= 0; k < channel; ++k){
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                img.data[count++] = data[i*step + j*channel + k]/255.;
            }
        }
    }
    rgbgr_image(img);
    PyObject *ret = pydetect(img, NULL, threshold);
    PyBuffer_Release(&buffer);

    return ret;
}


static PyMethodDef pyMethods[] =
{
      {"load", libpydarknet_load, METH_VARARGS, "load"},
      {"detect_file", libpydarknet_detect_file, METH_VARARGS, "detect_file"},
      {"detect", libpydarknet_detect, METH_VARARGS, "detect"},
      {NULL, NULL}
};

PyMODINIT_FUNC
initlibpydarknet(void)
{
    PyObject *m;
    
    m = Py_InitModule("libpydarknet", pyMethods);
    if (m == NULL)
        return;
}

