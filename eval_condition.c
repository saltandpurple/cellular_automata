#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include <math.h>

static PyMethodDef EvalConditionMethods[] = {
  { NULL, NULL, 0, NULL}
};


// Step 0
// If the cell has between 0 and 17 neighbours, it dies.
// If the cell has between 40 and 42 neighbours, it lives/spawns.
static short condition_step0(char state, char neighbours){
    return ((state == 1 && !(0 <= neighbours && neighbours <= 17)) || (40 <= neighbours && neighbours <= 42)) ? 1 : 0;
}

// Step 1
// If the cell has between 10 and 13 neighbours, it lives/spawns.
static char condition_step1(char state, char neighbours){
    return (state == 1 || (10 <= neighbours && neighbours <= 13)) ? 1 : 0;
}

// Step 2
// If the cell has between 9 and 21 neighbours, it dies.
static char condition_step2(char state, char neighbours){
    return (state == 0 || (9 <= neighbours && neighbours <= 21)) ? 0 : 1;
}

// Step 3
// If the cell has between 78 and 89 neighbours, it dies.
// If the cell has more than 108 neighbours, it dies.
static char condition_step3(char state, char neighbours){
    return  (state == 0 || (78 <= neighbours && neighbours <= 89) || 108 < neighbours) ? 0 : 1;
}

// in1: state of the cell
// in2: number of neighbours
// in3: current condition step
static void evalcondition(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data){
//  npy_intp i;
//  npy_intp n = dimensions[0];
  char *in1 = args[0], *in2 = args[1], *in3 = args[2];
  char *out1 = args[3];
  npy_intp in1_step = steps[0], in2_step = steps[1], in3_step = steps[2];
  npy_intp out1_step = steps[3];


  // Which step are we at?
  if (*in3 == 0) {
      *out1 = condition_step0(*in1, *in2);
  }
  else if (*in3 == 1) {
      *out1 = condition_step1(*in1, *in2);
  }
  else if (*in3 == 2) {
      *out1 = condition_step2(*in1, *in2);
  }
  else {
     *out1 = condition_step3(*in1, *in2);
  }
  //   Move the pointers to the next array element
  in1 += 16;
  in2 += 16;
  out1 += 16;
//  in1 += in1_step;
//  in2 += in2_step;
//  in3 += in3_step;
//  out1 += out1_step;
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&evalcondition};

/* These are the input and return dtypes of evalcondition.*/
// TODO: adjust these
static char types[4] = {NPY_BYTE, NPY_BYTE, NPY_BYTE, NPY_BYTE};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "evalcondition",
        NULL,
        -1,
        EvalConditionMethods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_evalcondition(void)
{
    PyObject *m, *evalcondition, *d;

    import_array();
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    evalcondition = PyUFunc_FromFuncAndData(funcs, NULL, types, 1, 3, 1,
                                    PyUFunc_None, "evalcondition",
                                    "evalcondition_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "evalcondition", evalcondition);
    Py_DECREF(evalcondition);

    return m;
}
