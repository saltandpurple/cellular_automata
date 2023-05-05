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
static short condition_step0(short state, short neighbours){
    return ((state == 1 && !(0 <= neighbours && neighbours <= 17)) || (40 <= neighbours && neighbours <= 42)) ? 1 : 0;
}

// Step 1
// If the cell has between 10 and 13 neighbours, it lives/spawns.
static short condition_step1(short state, short neighbours){
    return (state == 1 || (10 <= neighbours && neighbours <= 13)) ? 1 : 0;
}

// Step 2
// If the cell has between 9 and 21 neighbours, it dies.
static short condition_step2(short state, short neighbours){
    return (state == 0 || (9 <= neighbours && neighbours <= 21)) ? 0 : 1;
}

// Step 3
// If the cell has between 78 and 89 neighbours, it dies.
// If the cell has more than 108 neighbours, it dies.
static short condition_step3(short state, short neighbours){
    return  (state == 0 || (78 <= neighbours && neighbours <= 89) || 108 < neighbours) ? 0 : 1;
}

// in1: state of the cell
// in2: number of neighbours
// in3: current condition step
static void eval_condition(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data){
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in1 = args[0], *in2 = args[1], *in3 = args[2];
  char *out1 = args[3], *out2 = args[4];
  npy_intp in1_step = steps[0], in2_step = steps[1], in3_step = steps[2];
  npy_intp out1_step = steps[3], out2_step = steps[4];

  short tmp;

  // Which step are we at?
  if (in3 == 0) {
      out1 = condition_step0(in1, in2);
  }
  else if (in3 == 1) {
      out1 = condition_step1(in1, in2);
  }
  else if (in3 == 2) {
      out1 = condition_step2(in1, in2);
  }
  else {
      out1 = condition_step3(in1, in2);
  }

  in1 += in1_step;
  in2 += in2_step;
  in3 += in3_step;
  out1 += out1_step;
  out2 += out2_step;
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&evalcondition};

/* These are the input and return dtypes of eval_condition.*/
// TODO: adjust these
static char types[5] = {NPY_SHORT, NPY_SHORT, NPY_SHORT,
                        NPY_SHORT, NPY_SHORT};

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

    eval_condition = PyUFunc_FromFuncAndData(funcs, NULL, types, 1, 2, 2,
                                    PyUFunc_None, "evalcondition",
                                    "evalcondition_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "evalcondition", eval_condition);
    Py_DECREF(eval_condition);

    return m;
}
