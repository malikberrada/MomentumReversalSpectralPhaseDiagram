#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_21_API_VERSION
#include <numpy/arrayobject.h>

#include "native_api.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

namespace {
PyObject* py_feature_bank_batch(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject *z_obj=nullptr,*spans_obj=nullptr,*horizons_obj=nullptr;
    int spectral_window=0,min_periods=0;
    const char* backend="auto";
    static const char* kwlist[]={"z","spans","spectral_window","min_periods","horizons","backend",nullptr};
    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"OOiiO|s",const_cast<char**>(kwlist),
                                    &z_obj,&spans_obj,&spectral_window,&min_periods,&horizons_obj,&backend)) return nullptr;

    PyArrayObject* z=(PyArrayObject*)PyArray_FROM_OTF(z_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
    PyArrayObject* spans=(PyArrayObject*)PyArray_FROM_OTF(spans_obj,NPY_INT32,NPY_ARRAY_IN_ARRAY);
    PyArrayObject* horizons=(PyArrayObject*)PyArray_FROM_OTF(horizons_obj,NPY_INT32,NPY_ARRAY_IN_ARRAY);
    if(!z||!spans||!horizons){Py_XDECREF(z);Py_XDECREF(spans);Py_XDECREF(horizons);return nullptr;}
    if(PyArray_NDIM(z)!=2||PyArray_NDIM(spans)!=1||PyArray_NDIM(horizons)!=1){
        PyErr_SetString(PyExc_ValueError,"z must be 2D; spans and horizons must be 1D");
        Py_DECREF(z);Py_DECREF(spans);Py_DECREF(horizons);return nullptr;
    }
    if(spectral_window<=1||min_periods<=1||min_periods>spectral_window){
        PyErr_SetString(PyExc_ValueError,"invalid spectral_window/min_periods");
        Py_DECREF(z);Py_DECREF(spans);Py_DECREF(horizons);return nullptr;
    }
    const std::size_t B=(std::size_t)PyArray_DIM(z,0),T=(std::size_t)PyArray_DIM(z,1);
    const std::size_t S=(std::size_t)PyArray_DIM(spans,0),H=(std::size_t)PyArray_DIM(horizons,0);
    if(B==0||T==0||S==0||H==0){
        PyErr_SetString(PyExc_ValueError,"empty dimensions are not allowed");
        Py_DECREF(z);Py_DECREF(spans);Py_DECREF(horizons);return nullptr;
    }
    npy_intp psi_dims[3]={(npy_intp)B,(npy_intp)S,(npy_intp)T};
    npy_intp sum_dims[3]={(npy_intp)B,(npy_intp)H,(npy_intp)T};
    PyArrayObject* psi=(PyArrayObject*)PyArray_SimpleNew(3,psi_dims,NPY_DOUBLE);
    PyArrayObject* past=(PyArrayObject*)PyArray_SimpleNew(3,sum_dims,NPY_DOUBLE);
    PyArrayObject* future=(PyArrayObject*)PyArray_SimpleNew(3,sum_dims,NPY_DOUBLE);
    if(!psi||!past||!future){
        Py_XDECREF(psi);Py_XDECREF(past);Py_XDECREF(future);Py_DECREF(z);Py_DECREF(spans);Py_DECREF(horizons);return nullptr;
    }

    std::string chosen=backend?backend:"auto";
    if(chosen=="auto"){
#ifdef MRSPD_WITH_CUDA
        chosen=(mrspd_cuda_available()&&B*T>=200000)?"cuda":"cpu";
#else
        chosen="cpu";
#endif
    }
    std::string error;
    PyThreadState* save=PyEval_SaveThread();
    try {
        if(chosen=="cpu"){
            feature_bank_cpu((const double*)PyArray_DATA(z),B,T,(const int32_t*)PyArray_DATA(spans),S,
                             spectral_window,min_periods,(const int32_t*)PyArray_DATA(horizons),H,
                             (double*)PyArray_DATA(psi),(double*)PyArray_DATA(past),(double*)PyArray_DATA(future));
        } else if(chosen=="cuda") {
#ifdef MRSPD_WITH_CUDA
            feature_bank_cuda((const double*)PyArray_DATA(z),B,T,(const int32_t*)PyArray_DATA(spans),S,
                              spectral_window,min_periods,(const int32_t*)PyArray_DATA(horizons),H,
                              (double*)PyArray_DATA(psi),(double*)PyArray_DATA(past),(double*)PyArray_DATA(future));
#else
            error="mrspd-native was built without CUDA support. Reinstall with MRSPD_NATIVE_CUDA=1 after installing CUDA Toolkit/nvcc.";
#endif
        } else error="backend must be one of: auto, cpu, cuda";
    } catch(const std::exception& e){ error=e.what(); }
      catch(...){ error="unknown native exception"; }
    PyEval_RestoreThread(save);

    Py_DECREF(z);Py_DECREF(spans);Py_DECREF(horizons);
    if(!error.empty()){
        Py_DECREF(psi);Py_DECREF(past);Py_DECREF(future);
        PyErr_SetString(PyExc_RuntimeError,error.c_str());return nullptr;
    }
    PyObject* tup=PyTuple_New(3);
    PyTuple_SET_ITEM(tup,0,(PyObject*)psi);
    PyTuple_SET_ITEM(tup,1,(PyObject*)past);
    PyTuple_SET_ITEM(tup,2,(PyObject*)future);
    return tup;
}

PyObject* py_cuda_compiled(PyObject*,PyObject*){
#ifdef MRSPD_WITH_CUDA
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}
PyObject* py_cuda_available(PyObject*,PyObject*){
#ifdef MRSPD_WITH_CUDA
    if(mrspd_cuda_available())Py_RETURN_TRUE; else Py_RETURN_FALSE;
#else
    Py_RETURN_FALSE;
#endif
}

PyMethodDef methods[]={
    {"feature_bank_batch",(PyCFunction)(void(*)(void))py_feature_bank_batch,METH_VARARGS|METH_KEYWORDS,"Compute batched spectral and horizon features."},
    {"cuda_compiled",py_cuda_compiled,METH_NOARGS,"Whether CUDA kernels were compiled."},
    {"cuda_available",py_cuda_available,METH_NOARGS,"Whether a CUDA device is available."},
    {nullptr,nullptr,0,nullptr}
};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_core","MRSPD native C++/CUDA kernels",-1,methods};
}

PyMODINIT_FUNC PyInit__core(void){
    PyObject* m=PyModule_Create(&module);
    if(!m)return nullptr;
    if(_import_array()<0){Py_DECREF(m);return nullptr;}
    return m;
}
