#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include <math.h>

static PyMethodDef EvalConditionMethods[] = {
  { NULL, NULL, 0, NULL}
};

static void short_int_eval_conditions(char **args, cons npy_intp *dimensions, const npy_intp *steps, void *data){
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in1 = args[0], *in2 = args[1], *in3 = args[2];
  char *out1 = args[3], *out2 = args[4];
  npy_intp in1_step = steps[0], in2_step = steps[1];
  npy_intp out1_step = steps[2], out2_step = steps[3];

  short int tmp;

  for (i = 0; i < n; i++){

  }
}

// Step 0
// If the cell has between 0 and 17 neighbours, it dies.
// If the cell has between 40 and 42 neighbours, it lives/spawns.
static short int condition_step0(short int state, short int neighbours){
    if (0 <= neighbours && neighbours <= 17){
        return 0;
    }
    else if (state == 1 || (40 <= neighbours && neighbours <= 42)){
        return 1;
    }
    return 0;
}

// Step 1
// If the cell has between 10 and 13 neighbours, it lives/spawns.
static short int condition_step1(short int state, short int neighbours){
  if (state == 1 || (10 <= neighbours && neighbours <= 13)){
      return 1;
  }
  return 0;
}

// Step 2
// If the cell has between 9 and 21 neighbours, it dies.
static short int condition_step2(short int state, short int neighbours){
    if ()
}