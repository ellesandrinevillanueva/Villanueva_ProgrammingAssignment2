# ECE-2112-PA-2
##### Villanueva, Elle Sandrine P.  |  2ECE-C  |  Date Submitted: September 3, 2026

This document contains Programming Assignment 2 for the course Advanced Computer Programming, S.Y. 2026-2027. The objectives are to: 

1. Create and reshape NumPy arrays using NumPy functions;
2. Perform vectorized numerical operations on an ndarray;
3. Compute array statistics and use Boolean conditions to select elements; and
4. Save computed NumPy arrays as .npy files.

## A. Reproducible Normalization Problem

``import numpy as np``

This line imports the NumPy library in Python.

```
np.random.seed(2112)
X = np.random.randint(10, 101, size=(5, 5))
X

array([[48, 11, 15, 67, 21],
      [11, 41, 13, 66, 24],
      [71, 79, 53, 67, 70],
      [77, 35, 91, 19, 96],
      [35, 54, 37, 41, 17]], dtype=int32)
```
This section creates a 5x5 array of random numbers from 10 to 100.

```
mean = np.mean(X)
mean

np.float64(46.46)
```
The overall mean of the array is calculated.

```
standard_deviation = np.std(X)
standard_deviation

np.float64(25.864075471588002)
```
This function calculates the standard deviation of the array.

```
x_normalized = (X - mean) / standard_deviation
x_normalized

array([[ 0.06340841, -1.36714726, -1.2124926 ,  0.79801809, -0.98051059],
       [-1.36714726, -0.20723725, -1.28981993,  0.75935442, -0.86451959],
       [ 0.95267275,  1.26198209,  0.25672675,  0.79801809,  0.91400909],
       [ 1.18465476, -0.43921926,  1.72594609, -1.05783793,  1.91926443],
       [-0.43921926,  0.29539042, -0.36189192, -0.20723725, -1.13516526]])
```
x_normalized contains the normalized array using the mean and standard deviation.
Normalize the array using the mean and standard deviation.

```
np.mean(x_normalized)

np.float64(0.0)
```
This verifies that the mean is approximately 0.

```
np.std(x_normalized)
 
np.float64(0.9999999999999999)
```
This verifies that the standard deviation is approximately 1.

``np.save("X_normalized.npy", x_normalized)``
This saves the normalized array as a .npy file.

## B. Cubes Divisible By 4 Problem"

```
C = np.arange(1,101,1)
C

array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
      27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
      40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
      53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
      66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
      79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
      92,  93,  94,  95,  96,  97,  98,  99, 100])
```
``np.arange`` creates a one-dimensional array that contains integers from 1-100.

```
new_c = C ** 3
      new_c

array([      1,       8,      27,      64,     125,     216,     343,\n",
      512,     729,    1000,    1331,    1728,    2197,    2744,
      3375,    4096,    4913,    5832,    6859,    8000,    9261,
      10648,   12167,   13824,   15625,   17576,   19683,   21952,
      24389,   27000,   29791,   32768,   35937,   39304,   42875,
      46656,   50653,   54872,   59319,   64000,   68921,   74088,
      79507,   85184,   91125,   97336,  103823,  110592,  117649,
      125000,  132651,  140608,  148877,  157464,  166375,  175616,
      185193,  195112,  205379,  216000,  226981,  238328,  250047,
      262144,  274625,  287496,  300763,  314432,  328509,  343000,
      357911,  373248,  389017,  405224,  421875,  438976,  456533,
      474552,  493039,  512000,  531441,  551368,  571787,  592704,
      614125,  636056,  658503,  681472,  704969,  729000,  753571,
      778688,  804357,  830584,  857375,  884736,  912673,  941192,
      970299, 1000000])
```
``new_c`` stores each element of the C array to the third power.

```
new_c = new_c.reshape(10,10)
    new_c

array([[      1,       8,      27,      64,     125,     216,     343,
            512,     729,    1000],
      [   1331,    1728,    2197,    2744,    3375,    4096,    4913,
            5832,    6859,    8000],
      [   9261,   10648,   12167,   13824,   15625,   17576,   19683,
            21952,   24389,   27000],
      [  29791,   32768,   35937,   39304,   42875,   46656,   50653,
            54872,   59319,   64000],
      [  68921,   74088,   79507,   85184,   91125,   97336,  103823,
            110592,  117649,  125000],
      [ 132651,  140608,  148877,  157464,  166375,  175616,  185193,
            195112,  205379,  216000],
      [ 226981,  238328,  250047,  262144,  274625,  287496,  300763,
            314432,  328509,  343000],
      [ 357911,  373248,  389017,  405224,  421875,  438976,  456533,
            474552,  493039,  512000],
      [ 531441,  551368,  571787,  592704,  614125,  636056,  658503,
            681472,  704969,  729000],
      [ 753571,  778688,  804357,  830584,  857375,  884736,  912673,
            941192,  970299, 1000000]])
```


```
array([[False,  True, False,  True, False,  True, False,  True, False,
      True],
      [False,  True, False,  True, False,  True, False,  True, False,
      True],
      [False,  True, False,  True, False,  True, False,  True, False,
      True],
      [False,  True, False,  True, False,  True, False,  True, False,
      True],
      [False,  True, False,  True, False,  True, False,  True, False,
      True],
      [False,  True, False,  True, False,  True, False,  True, False,
      True],
      [False,  True, False,  True, False,  True, False,  True, False,
      True],
      [False,  True, False,  True, False,  True, False,  True, False,
      True],
      [False,  True, False,  True, False,  True, False,  True, False,
      True],
      [False,  True, False,  True, False,  True, False,  True, False,
      True]])
```

    "bool_c = new_c % 4 == 0\n",
    "bool_c"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c742a947-efeb-426c-8d9f-590e0a0938ad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([      8,      64,     216,     512,    1000,    1728,    2744,\n",
       "          4096,    5832,    8000,   10648,   13824,   17576,   21952,\n",
       "         27000,   32768,   39304,   46656,   54872,   64000,   74088,\n",
       "         85184,   97336,  110592,  125000,  140608,  157464,  175616,\n",
       "        195112,  216000,  238328,  262144,  287496,  314432,  343000,\n",
       "        373248,  405224,  438976,  474552,  512000,  551368,  592704,\n",
       "        636056,  681472,  729000,  778688,  830584,  884736,  941192,\n",
       "       1000000])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "div_by_4 = new_c[bool_c]\n",
    "div_by_4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b3a72fe6-bd40-43c0-99c1-afe18ce41313",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.save(\"div_by_4.npy\", div_by_4)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe0a0c5b-b2bd-492d-8228-20c671fe6396",
   "metadata": {},
   "source": [
    "# C. Above-Mean Squares Problem"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "473ad18d-ac65-44ee-8492-eb061769f87d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([   1,    4,    9,   16,   25,   36,   49,   64,   81,  100,  121,\n",
       "        144,  169,  196,  225,  256,  289,  324,  361,  400,  441,  484,\n",
       "        529,  576,  625,  676,  729,  784,  841,  900,  961, 1024, 1089,\n",
       "       1156, 1225, 1296])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "S = np.arange(1,37,1) ** 2\n",
    "S"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "50d15731-6813-4e57-ac12-5ec74e1e125f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[   1,    4,    9,   16,   25,   36],\n",
       "       [  49,   64,   81,  100,  121,  144],\n",
       "       [ 169,  196,  225,  256,  289,  324],\n",
       "       [ 361,  400,  441,  484,  529,  576],\n",
       "       [ 625,  676,  729,  784,  841,  900],\n",
       "       [ 961, 1024, 1089, 1156, 1225, 1296]])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "S = S.reshape(6,6)\n",
    "S"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "4ac2c4a9-0115-45b1-8139-ffeb66ab14c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.float64(450.1666666666667)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "S_mean = np.mean(S)\n",
    "S_mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "00ab41a0-c807-4531-b2ed-f6f0959cbb78",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[False, False, False, False, False, False],\n",
       "       [False, False, False, False, False, False],\n",
       "       [False, False, False, False, False, False],\n",
       "       [False, False, False,  True,  True,  True],\n",
       "       [ True,  True,  True,  True,  True,  True],\n",
       "       [ True,  True,  True,  True,  True,  True]])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bool_mean = S > S_mean\n",
    "bool_mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "73feb5e1-2aab-426c-8934-efdd61685ff3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 484,  529,  576,  625,  676,  729,  784,  841,  900,  961, 1024,\n",
       "       1089, 1156, 1225, 1296])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "above_mean = S[S > S_mean]\n",
    "above_mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "38ffb4c2-6e5b-422b-a524-13d75f3fd61c",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.save(\"above_mean.npy\", above_mean)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

README File Version History
September 3, 2026 - Initial submission
September 6, 2026 - Added descriptions
