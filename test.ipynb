{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:23:15.079234Z",
     "start_time": "2018-04-07T17:23:14.228909Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'\n",
    "n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))\n",
    "model_name = 'nasnet_fine_tuning_8clf_2'\n",
    "\n",
    "import keras.backend as K\n",
    "import tensorflow as tf\n",
    "\n",
    "# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)\n",
    "# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 导入必要的库"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:23:15.316079Z",
     "start_time": "2018-04-07T17:23:15.080390Z"
    }
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from keras.layers import *\n",
    "from keras.models import *\n",
    "from keras.optimizers import *\n",
    "from keras.applications import *\n",
    "from keras.regularizers import l2\n",
    "\n",
    "from keras.preprocessing.image import *\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import random\n",
    "import os\n",
    "import cv2\n",
    "from tqdm import tqdm\n",
    "from glob import glob\n",
    "import multiprocessing\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from collections import Counter\n",
    "from keras import backend as K\n",
    "from keras.utils import multi_gpu_model\n",
    "\n",
    "from IPython.display import display, Image\n",
    "\n",
    "%matplotlib inline\n",
    "%config InlineBackend.figure_format = 'retina'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 读取数据集"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:23:15.472275Z",
     "start_time": "2018-04-07T17:23:15.317186Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>filename</th>\n",
       "      <th>label_name</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Images/skirt_length_labels/0575dc5183059223924...</td>\n",
       "      <td>skirt_length</td>\n",
       "      <td>nynnnn</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Images/collar_design_labels/ba453aef46c484d4f3...</td>\n",
       "      <td>collar_design</td>\n",
       "      <td>nnynn</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Images/coat_length_labels/ac448ded8b3ebd88a998...</td>\n",
       "      <td>coat_length</td>\n",
       "      <td>nnnnnynn</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Images/coat_length_labels/206b38af4f2d55393865...</td>\n",
       "      <td>coat_length</td>\n",
       "      <td>nnnnnynn</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Images/pant_length_labels/7da34d442a8dcc7c84a7...</td>\n",
       "      <td>pant_length</td>\n",
       "      <td>nnynnn</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                            filename     label_name     label\n",
       "0  Images/skirt_length_labels/0575dc5183059223924...   skirt_length    nynnnn\n",
       "1  Images/collar_design_labels/ba453aef46c484d4f3...  collar_design     nnynn\n",
       "2  Images/coat_length_labels/ac448ded8b3ebd88a998...    coat_length  nnnnnynn\n",
       "3  Images/coat_length_labels/206b38af4f2d55393865...    coat_length  nnnnnynn\n",
       "4  Images/pant_length_labels/7da34d442a8dcc7c84a7...    pant_length    nnynnn"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "Counter({'coat_length': 11320,\n",
       "         'collar_design': 8393,\n",
       "         'lapel_design': 7034,\n",
       "         'neck_design': 5696,\n",
       "         'neckline_design': 17148,\n",
       "         'pant_length': 7460,\n",
       "         'skirt_length': 9223,\n",
       "         'sleeve_length': 13299})"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('base/Annotations/label.csv', header=None)\n",
    "df.columns = ['filename', 'label_name', 'label']\n",
    "df = df.sample(frac=1).reset_index(drop=True) # shuffle\n",
    "\n",
    "df.label_name = df.label_name.str.replace('_labels', '')\n",
    "\n",
    "display(df.head())\n",
    "c = Counter(df.label_name)\n",
    "c"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:23:15.523937Z",
     "start_time": "2018-04-07T17:23:15.474352Z"
    },
    "code_folding": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'coat_length': 8,\n",
       " 'collar_design': 5,\n",
       " 'lapel_design': 5,\n",
       " 'neck_design': 5,\n",
       " 'neckline_design': 10,\n",
       " 'pant_length': 6,\n",
       " 'skirt_length': 6,\n",
       " 'sleeve_length': 9}"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "label_count = dict([(x, len(df[df.label_name == x].label.values[0])) for x in c.keys()])\n",
    "label_names = list(label_count.keys())\n",
    "display(label_count)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 生成 y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:23:17.137384Z",
     "start_time": "2018-04-07T17:23:15.524971Z"
    }
   },
   "outputs": [],
   "source": [
    "fnames = df['filename'].values\n",
    "width = 331\n",
    "n = len(df)\n",
    "y = [np.zeros((n, label_count[x])) for x in label_count.keys()]\n",
    "for i in range(n):\n",
    "    label_name = df.label_name[i]\n",
    "    label = df.label[i]\n",
    "    y[label_names.index(label_name)][i, label.find('y')] = 1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 读取图片"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:24:15.086526Z",
     "start_time": "2018-04-07T17:23:17.140308Z"
    },
    "code_folding": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 79573/79573 [00:57<00:00, 1377.73it/s]\n"
     ]
    }
   ],
   "source": [
    "def f(index):\n",
    "    return index, cv2.resize(cv2.imread('base/'+fnames[index]), (width, width))\n",
    "\n",
    "X = np.zeros((n, width, width, 3), dtype=np.uint8)\n",
    "with multiprocessing.Pool(12) as pool:\n",
    "    with tqdm(pool.imap_unordered(f, range(n)), total=n) as pbar:\n",
    "        for i, img in pbar:\n",
    "            X[i] = img[:,:,::-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:24:15.093182Z",
     "start_time": "2018-04-07T17:24:15.088621Z"
    }
   },
   "outputs": [],
   "source": [
    "n_train = int(n*0.8)\n",
    "\n",
    "X_train = X[:n_train]\n",
    "X_valid = X[n_train:]\n",
    "y_train = [x[:n_train] for x in y]\n",
    "y_valid = [x[n_train:] for x in y]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:24:15.099505Z",
     "start_time": "2018-04-07T17:24:15.094585Z"
    }
   },
   "outputs": [],
   "source": [
    "def display_images(imgs, w=8, h=4, figsize=(24, 12)):\n",
    "    plt.figure(figsize=figsize)\n",
    "    for i in range(w*h):\n",
    "        plt.subplot(h, w, i+1)\n",
    "        plt.imshow(imgs[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:24:15.116015Z",
     "start_time": "2018-04-07T17:24:15.100880Z"
    }
   },
   "outputs": [],
   "source": [
    "class Generator():\n",
    "    def __init__(self, X, y, batch_size=32, aug=False):\n",
    "        def generator():\n",
    "            idg = ImageDataGenerator(horizontal_flip=True,\n",
    "                                     rotation_range=20,\n",
    "                                     zoom_range=0.2)\n",
    "            while True:\n",
    "                for i in range(0, len(X), batch_size):\n",
    "                    X_batch = X[i:i+batch_size].copy()\n",
    "                    y_barch = [x[i:i+batch_size] for x in y]\n",
    "                    if aug:\n",
    "                        for j in range(len(X_batch)):\n",
    "                            X_batch[j] = idg.random_transform(X_batch[j])\n",
    "                    yield X_batch, y_barch\n",
    "        self.generator = generator()\n",
    "        self.steps = len(X) // batch_size + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:24:15.123929Z",
     "start_time": "2018-04-07T17:24:15.117484Z"
    }
   },
   "outputs": [],
   "source": [
    "gen_train = Generator(X_train, y_train, batch_size=32, aug=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 数据集探索\n",
    "\n",
    "## 类别分布"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:24:15.130162Z",
     "start_time": "2018-04-07T17:24:15.125259Z"
    }
   },
   "outputs": [],
   "source": [
    "# plt.figure(figsize=(26, 14))\n",
    "# for i in range(8):\n",
    "#     plt.subplot(2, 4, i+1)\n",
    "#     counts = Counter(y[i].argmax(axis=-1)[np.where(y[i].any(axis=-1))])\n",
    "#     pd.Series(counts).plot('bar')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 抽样可视化"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:24:15.138224Z",
     "start_time": "2018-04-07T17:24:15.131739Z"
    }
   },
   "outputs": [],
   "source": [
    "# plt.figure(figsize=(26, 14))\n",
    "# w = 8\n",
    "# h = 4\n",
    "# for i in range(w*h):\n",
    "#     plt.subplot(h, w, i+1)\n",
    "#     index = np.random.randint(n)\n",
    "#     plt.title(str([y[x][index].argmax() if y[x][index].any() else -1 for x in range(8) ]))\n",
    "#     plt.imshow(X[index])\n",
    "#     plt.axis('off')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 搭建模型并训练\n",
    "\n",
    "## 搭建模型"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:24:15.145368Z",
     "start_time": "2018-04-07T17:24:15.139606Z"
    }
   },
   "outputs": [],
   "source": [
    "def acc(y_true, y_pred):\n",
    "    index = tf.reduce_any(y_true > 0.5, axis=-1)\n",
    "    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))\n",
    "    index = tf.cast(index, tf.float32)\n",
    "    res = tf.cast(res, tf.float32)\n",
    "    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:25:14.609365Z",
     "start_time": "2018-04-07T17:24:15.146696Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/svg+xml": [
       "<svg height=\"387pt\" viewBox=\"0.00 0.00 2312.50 387.00\" width=\"2313pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n",
       "<g class=\"graph\" id=\"graph0\" transform=\"scale(1 1) rotate(0) translate(4 383)\">\n",
       "<title>G</title>\n",
       "<polygon fill=\"white\" points=\"-4,4 -4,-383 2308.5,-383 2308.5,4 -4,4\" stroke=\"none\"/>\n",
       "<!-- 140213022640672 -->\n",
       "<g class=\"node\" id=\"node1\"><title>140213022640672</title>\n",
       "<polygon fill=\"none\" points=\"989.5,-332.5 989.5,-378.5 1294.5,-378.5 1294.5,-332.5 989.5,-332.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1052\" y=\"-351.8\">input_2: InputLayer</text>\n",
       "<polyline fill=\"none\" points=\"1114.5,-332.5 1114.5,-378.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1142\" y=\"-363.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"1114.5,-355.5 1169.5,-355.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1142\" y=\"-340.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"1169.5,-332.5 1169.5,-378.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1232\" y=\"-363.3\">(None, 331, 331, 3)</text>\n",
       "<polyline fill=\"none\" points=\"1169.5,-355.5 1294.5,-355.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1232\" y=\"-340.3\">(None, 331, 331, 3)</text>\n",
       "</g>\n",
       "<!-- 140207856588784 -->\n",
       "<g class=\"node\" id=\"node2\"><title>140207856588784</title>\n",
       "<polygon fill=\"none\" points=\"990.5,-249.5 990.5,-295.5 1293.5,-295.5 1293.5,-249.5 990.5,-249.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1052\" y=\"-268.8\">lambda_1: Lambda</text>\n",
       "<polyline fill=\"none\" points=\"1113.5,-249.5 1113.5,-295.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1141\" y=\"-280.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"1113.5,-272.5 1168.5,-272.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1141\" y=\"-257.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"1168.5,-249.5 1168.5,-295.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1231\" y=\"-280.3\">(None, 331, 331, 3)</text>\n",
       "<polyline fill=\"none\" points=\"1168.5,-272.5 1293.5,-272.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1231\" y=\"-257.3\">(None, 331, 331, 3)</text>\n",
       "</g>\n",
       "<!-- 140213022640672&#45;&gt;140207856588784 -->\n",
       "<g class=\"edge\" id=\"edge1\"><title>140213022640672-&gt;140207856588784</title>\n",
       "<path d=\"M1142,-332.366C1142,-324.152 1142,-314.658 1142,-305.725\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"1145.5,-305.607 1142,-295.607 1138.5,-305.607 1145.5,-305.607\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140207871784384 -->\n",
       "<g class=\"node\" id=\"node3\"><title>140207871784384</title>\n",
       "<polygon fill=\"none\" points=\"998.5,-166.5 998.5,-212.5 1285.5,-212.5 1285.5,-166.5 998.5,-166.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1052\" y=\"-185.8\">NASNet: Model</text>\n",
       "<polyline fill=\"none\" points=\"1105.5,-166.5 1105.5,-212.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1133\" y=\"-197.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"1105.5,-189.5 1160.5,-189.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1133\" y=\"-174.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"1160.5,-166.5 1160.5,-212.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1223\" y=\"-197.3\">(None, 331, 331, 3)</text>\n",
       "<polyline fill=\"none\" points=\"1160.5,-189.5 1285.5,-189.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1223\" y=\"-174.3\">(None, 4032)</text>\n",
       "</g>\n",
       "<!-- 140207856588784&#45;&gt;140207871784384 -->\n",
       "<g class=\"edge\" id=\"edge2\"><title>140207856588784-&gt;140207871784384</title>\n",
       "<path d=\"M1142,-249.366C1142,-241.152 1142,-231.658 1142,-222.725\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"1145.5,-222.607 1142,-212.607 1138.5,-222.607 1145.5,-222.607\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140207856704648 -->\n",
       "<g class=\"node\" id=\"node4\"><title>140207856704648</title>\n",
       "<polygon fill=\"none\" points=\"1007.5,-83.5 1007.5,-129.5 1276.5,-129.5 1276.5,-83.5 1007.5,-83.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1070\" y=\"-102.8\">dropout_1: Dropout</text>\n",
       "<polyline fill=\"none\" points=\"1132.5,-83.5 1132.5,-129.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1160\" y=\"-114.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"1132.5,-106.5 1187.5,-106.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1160\" y=\"-91.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"1187.5,-83.5 1187.5,-129.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1232\" y=\"-114.3\">(None, 4032)</text>\n",
       "<polyline fill=\"none\" points=\"1187.5,-106.5 1276.5,-106.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1232\" y=\"-91.3\">(None, 4032)</text>\n",
       "</g>\n",
       "<!-- 140207871784384&#45;&gt;140207856704648 -->\n",
       "<g class=\"edge\" id=\"edge3\"><title>140207871784384-&gt;140207856704648</title>\n",
       "<path d=\"M1142,-166.366C1142,-158.152 1142,-148.658 1142,-139.725\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"1145.5,-139.607 1142,-129.607 1138.5,-139.607 1145.5,-139.607\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140207851978936 -->\n",
       "<g class=\"node\" id=\"node5\"><title>140207851978936</title>\n",
       "<polygon fill=\"none\" points=\"0,-0.5 0,-46.5 266,-46.5 266,-0.5 0,-0.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"61\" y=\"-19.8\">skirt_length: Dense</text>\n",
       "<polyline fill=\"none\" points=\"122,-0.5 122,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"149.5\" y=\"-31.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"122,-23.5 177,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"149.5\" y=\"-8.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"177,-0.5 177,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"221.5\" y=\"-31.3\">(None, 4032)</text>\n",
       "<polyline fill=\"none\" points=\"177,-23.5 266,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"221.5\" y=\"-8.3\">(None, 6)</text>\n",
       "</g>\n",
       "<!-- 140207856704648&#45;&gt;140207851978936 -->\n",
       "<g class=\"edge\" id=\"edge4\"><title>140207856704648-&gt;140207851978936</title>\n",
       "<path d=\"M1007.25,-100.16C838.641,-92.6283 540.904,-76.5259 276.542,-47.1286\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"276.688,-43.6231 266.361,-45.9879 275.909,-50.5796 276.688,-43.6231\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140207855492232 -->\n",
       "<g class=\"node\" id=\"node6\"><title>140207855492232</title>\n",
       "<polygon fill=\"none\" points=\"284.5,-0.5 284.5,-46.5 559.5,-46.5 559.5,-0.5 284.5,-0.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"350\" y=\"-19.8\">collar_design: Dense</text>\n",
       "<polyline fill=\"none\" points=\"415.5,-0.5 415.5,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"443\" y=\"-31.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"415.5,-23.5 470.5,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"443\" y=\"-8.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"470.5,-0.5 470.5,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"515\" y=\"-31.3\">(None, 4032)</text>\n",
       "<polyline fill=\"none\" points=\"470.5,-23.5 559.5,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"515\" y=\"-8.3\">(None, 5)</text>\n",
       "</g>\n",
       "<!-- 140207856704648&#45;&gt;140207855492232 -->\n",
       "<g class=\"edge\" id=\"edge5\"><title>140207856704648-&gt;140207855492232</title>\n",
       "<path d=\"M1007.24,-93.7355C893.279,-83.3261 725.429,-66.9167 569.761,-47.1989\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"569.888,-43.6868 559.526,-45.8953 569.003,-50.6307 569.888,-43.6868\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140207824232632 -->\n",
       "<g class=\"node\" id=\"node7\"><title>140207824232632</title>\n",
       "<polygon fill=\"none\" points=\"577.5,-0.5 577.5,-46.5 842.5,-46.5 842.5,-0.5 577.5,-0.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"638\" y=\"-19.8\">coat_length: Dense</text>\n",
       "<polyline fill=\"none\" points=\"698.5,-0.5 698.5,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"726\" y=\"-31.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"698.5,-23.5 753.5,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"726\" y=\"-8.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"753.5,-0.5 753.5,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"798\" y=\"-31.3\">(None, 4032)</text>\n",
       "<polyline fill=\"none\" points=\"753.5,-23.5 842.5,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"798\" y=\"-8.3\">(None, 8)</text>\n",
       "</g>\n",
       "<!-- 140207856704648&#45;&gt;140207824232632 -->\n",
       "<g class=\"edge\" id=\"edge6\"><title>140207856704648-&gt;140207824232632</title>\n",
       "<path d=\"M1024.52,-83.4734C966.659,-72.6235 896.787,-59.5225 837.61,-48.4269\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"837.926,-44.9251 827.452,-46.5222 836.636,-51.8052 837.926,-44.9251\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140207114422704 -->\n",
       "<g class=\"node\" id=\"node8\"><title>140207114422704</title>\n",
       "<polygon fill=\"none\" points=\"861,-0.5 861,-46.5 1127,-46.5 1127,-0.5 861,-0.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"922\" y=\"-19.8\">pant_length: Dense</text>\n",
       "<polyline fill=\"none\" points=\"983,-0.5 983,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1010.5\" y=\"-31.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"983,-23.5 1038,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1010.5\" y=\"-8.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"1038,-0.5 1038,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1082.5\" y=\"-31.3\">(None, 4032)</text>\n",
       "<polyline fill=\"none\" points=\"1038,-23.5 1127,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1082.5\" y=\"-8.3\">(None, 6)</text>\n",
       "</g>\n",
       "<!-- 140207856704648&#45;&gt;140207114422704 -->\n",
       "<g class=\"edge\" id=\"edge7\"><title>140207856704648-&gt;140207114422704</title>\n",
       "<path d=\"M1101.56,-83.3664C1083.53,-73.4998 1062.13,-61.7881 1043.19,-51.4194\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"1044.85,-48.3376 1034.39,-46.6068 1041.48,-54.4781 1044.85,-48.3376\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140206937666000 -->\n",
       "<g class=\"node\" id=\"node9\"><title>140206937666000</title>\n",
       "<polygon fill=\"none\" points=\"1145,-0.5 1145,-46.5 1435,-46.5 1435,-0.5 1145,-0.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1218\" y=\"-19.8\">neckline_design: Dense</text>\n",
       "<polyline fill=\"none\" points=\"1291,-0.5 1291,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1318.5\" y=\"-31.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"1291,-23.5 1346,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1318.5\" y=\"-8.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"1346,-0.5 1346,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1390.5\" y=\"-31.3\">(None, 4032)</text>\n",
       "<polyline fill=\"none\" points=\"1346,-23.5 1435,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1390.5\" y=\"-8.3\">(None, 10)</text>\n",
       "</g>\n",
       "<!-- 140207856704648&#45;&gt;140206937666000 -->\n",
       "<g class=\"edge\" id=\"edge8\"><title>140207856704648-&gt;140206937666000</title>\n",
       "<path d=\"M1182.44,-83.3664C1200.47,-73.4998 1221.87,-61.7881 1240.81,-51.4194\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"1242.52,-54.4781 1249.61,-46.6068 1239.15,-48.3376 1242.52,-54.4781\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140206937232944 -->\n",
       "<g class=\"node\" id=\"node10\"><title>140206937232944</title>\n",
       "<polygon fill=\"none\" points=\"1453,-0.5 1453,-46.5 1723,-46.5 1723,-0.5 1453,-0.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1516\" y=\"-19.8\">lapel_design: Dense</text>\n",
       "<polyline fill=\"none\" points=\"1579,-0.5 1579,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1606.5\" y=\"-31.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"1579,-23.5 1634,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1606.5\" y=\"-8.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"1634,-0.5 1634,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1678.5\" y=\"-31.3\">(None, 4032)</text>\n",
       "<polyline fill=\"none\" points=\"1634,-23.5 1723,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1678.5\" y=\"-8.3\">(None, 5)</text>\n",
       "</g>\n",
       "<!-- 140207856704648&#45;&gt;140206937232944 -->\n",
       "<g class=\"edge\" id=\"edge9\"><title>140207856704648-&gt;140206937232944</title>\n",
       "<path d=\"M1263.28,-83.4734C1323.15,-72.6011 1395.46,-59.4683 1456.63,-48.3582\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"1457.53,-51.7529 1466.74,-46.5222 1456.28,-44.8656 1457.53,-51.7529\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140206937333376 -->\n",
       "<g class=\"node\" id=\"node11\"><title>140206937333376</title>\n",
       "<polygon fill=\"none\" points=\"1741,-0.5 1741,-46.5 2017,-46.5 2017,-0.5 1741,-0.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1807\" y=\"-19.8\">sleeve_length: Dense</text>\n",
       "<polyline fill=\"none\" points=\"1873,-0.5 1873,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1900.5\" y=\"-31.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"1873,-23.5 1928,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1900.5\" y=\"-8.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"1928,-0.5 1928,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1972.5\" y=\"-31.3\">(None, 4032)</text>\n",
       "<polyline fill=\"none\" points=\"1928,-23.5 2017,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"1972.5\" y=\"-8.3\">(None, 9)</text>\n",
       "</g>\n",
       "<!-- 140207856704648&#45;&gt;140206937333376 -->\n",
       "<g class=\"edge\" id=\"edge10\"><title>140207856704648-&gt;140206937333376</title>\n",
       "<path d=\"M1276.67,-94.1952C1393.95,-83.8729 1568.79,-67.3372 1730.77,-47.1263\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"1731.44,-50.5697 1740.92,-45.852 1730.57,-43.6242 1731.44,-50.5697\" stroke=\"black\"/>\n",
       "</g>\n",
       "<!-- 140206937027248 -->\n",
       "<g class=\"node\" id=\"node12\"><title>140206937027248</title>\n",
       "<polygon fill=\"none\" points=\"2035.5,-0.5 2035.5,-46.5 2304.5,-46.5 2304.5,-0.5 2035.5,-0.5\" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"2098\" y=\"-19.8\">neck_design: Dense</text>\n",
       "<polyline fill=\"none\" points=\"2160.5,-0.5 2160.5,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"2188\" y=\"-31.3\">input:</text>\n",
       "<polyline fill=\"none\" points=\"2160.5,-23.5 2215.5,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"2188\" y=\"-8.3\">output:</text>\n",
       "<polyline fill=\"none\" points=\"2215.5,-0.5 2215.5,-46.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"2260\" y=\"-31.3\">(None, 4032)</text>\n",
       "<polyline fill=\"none\" points=\"2215.5,-23.5 2304.5,-23.5 \" stroke=\"black\"/>\n",
       "<text font-family=\"Times,serif\" font-size=\"14.00\" text-anchor=\"middle\" x=\"2260\" y=\"-8.3\">(None, 5)</text>\n",
       "</g>\n",
       "<!-- 140207856704648&#45;&gt;140206937027248 -->\n",
       "<g class=\"edge\" id=\"edge11\"><title>140207856704648-&gt;140206937027248</title>\n",
       "<path d=\"M1276.74,-100.312C1448.24,-92.8651 1753.77,-76.7733 2025.1,-47.0516\" fill=\"none\" stroke=\"black\"/>\n",
       "<polygon fill=\"black\" points=\"2025.59,-50.5192 2035.14,-45.9435 2024.82,-43.5614 2025.59,-50.5192\" stroke=\"black\"/>\n",
       "</g>\n",
       "</g>\n",
       "</svg>"
      ],
      "text/plain": [
       "<IPython.core.display.SVG object>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "base_model = NASNetLarge(weights='imagenet', input_shape=(width, width, 3), include_top=False, pooling='avg')\n",
    "\n",
    "input_tensor = Input((width, width, 3))\n",
    "x = input_tensor\n",
    "x = Lambda(nasnet.preprocess_input)(x)\n",
    "x = base_model(x)\n",
    "x = Dropout(0.5)(x)\n",
    "x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()]\n",
    "\n",
    "model = Model(input_tensor, x)\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])\n",
    "\n",
    "from IPython.display import SVG\n",
    "from keras.utils.vis_utils import model_to_dot, plot_model\n",
    "\n",
    "plot_model(model, show_shapes=True, to_file='model_simple.png')\n",
    "SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:27:36.541858Z",
     "start_time": "2018-04-07T17:25:14.610710Z"
    }
   },
   "outputs": [],
   "source": [
    "model2 = multi_gpu_model(model, n_gpus)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T17:27:36.545662Z",
     "start_time": "2018-04-07T17:27:36.543513Z"
    }
   },
   "outputs": [],
   "source": [
    "# opt = SGD(1e-3, momentum=0.9, nesterov=True, decay=1e-5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T19:26:22.717817Z",
     "start_time": "2018-04-07T17:27:36.547216Z"
    },
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "1990/1990 [==============================] - 3637s 2s/step - loss: 0.9191 - skirt_length_loss: 0.0899 - collar_design_loss: 0.0950 - coat_length_loss: 0.1324 - pant_length_loss: 0.0665 - neckline_design_loss: 0.2276 - lapel_design_loss: 0.0754 - sleeve_length_loss: 0.1565 - neck_design_loss: 0.0758 - skirt_length_acc: 0.7035 - collar_design_acc: 0.6344 - coat_length_acc: 0.6599 - pant_length_acc: 0.6951 - neckline_design_acc: 0.6305 - lapel_design_acc: 0.6307 - sleeve_length_acc: 0.6449 - neck_design_acc: 0.5332 - val_loss: 0.6083 - val_skirt_length_loss: 0.0512 - val_collar_design_loss: 0.0676 - val_coat_length_loss: 0.0879 - val_pant_length_loss: 0.0449 - val_neckline_design_loss: 0.1584 - val_lapel_design_loss: 0.0478 - val_sleeve_length_loss: 0.0960 - val_neck_design_loss: 0.0545 - val_skirt_length_acc: 0.5910 - val_collar_design_acc: 0.5061 - val_coat_length_acc: 0.6075 - val_pant_length_acc: 0.5270 - val_neckline_design_acc: 0.6930 - val_lapel_design_acc: 0.4736 - val_sleeve_length_acc: 0.6544 - val_neck_design_acc: 0.3755\n",
      "Epoch 2/2\n",
      "1990/1990 [==============================] - 3440s 2s/step - loss: 0.5394 - skirt_length_loss: 0.0565 - collar_design_loss: 0.0582 - coat_length_loss: 0.0797 - pant_length_loss: 0.0391 - neckline_design_loss: 0.1297 - lapel_design_loss: 0.0422 - sleeve_length_loss: 0.0853 - neck_design_loss: 0.0489 - skirt_length_acc: 0.8123 - collar_design_acc: 0.7756 - coat_length_acc: 0.7939 - pant_length_acc: 0.8053 - neckline_design_acc: 0.7990 - lapel_design_acc: 0.7909 - sleeve_length_acc: 0.8148 - neck_design_acc: 0.6859 - val_loss: 0.5533 - val_skirt_length_loss: 0.0458 - val_collar_design_loss: 0.0624 - val_coat_length_loss: 0.0826 - val_pant_length_loss: 0.0409 - val_neckline_design_loss: 0.1371 - val_lapel_design_loss: 0.0490 - val_sleeve_length_loss: 0.0908 - val_neck_design_loss: 0.0448 - val_skirt_length_acc: 0.6060 - val_collar_design_acc: 0.5263 - val_coat_length_acc: 0.6107 - val_pant_length_acc: 0.5369 - val_neckline_design_acc: 0.7293 - val_lapel_design_acc: 0.4752 - val_sleeve_length_acc: 0.6794 - val_neck_design_acc: 0.3972\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f842eb77e48>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=[acc])\n",
    "model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=2, validation_data=(X_valid, y_valid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T22:02:58.261416Z",
     "start_time": "2018-04-07T19:26:22.719422Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/3\n",
      "1990/1990 [==============================] - 3266s 2s/step - loss: 0.3240 - skirt_length_loss: 0.0344 - collar_design_loss: 0.0373 - coat_length_loss: 0.0482 - pant_length_loss: 0.0250 - neckline_design_loss: 0.0757 - lapel_design_loss: 0.0225 - sleeve_length_loss: 0.0507 - neck_design_loss: 0.0301 - skirt_length_acc: 0.8773 - collar_design_acc: 0.8431 - coat_length_acc: 0.8804 - pant_length_acc: 0.8643 - neckline_design_acc: 0.8781 - lapel_design_acc: 0.8650 - sleeve_length_acc: 0.8903 - neck_design_acc: 0.7746 - val_loss: 0.4108 - val_skirt_length_loss: 0.0391 - val_collar_design_loss: 0.0455 - val_coat_length_loss: 0.0607 - val_pant_length_loss: 0.0323 - val_neckline_design_loss: 0.1020 - val_lapel_design_loss: 0.0334 - val_sleeve_length_loss: 0.0597 - val_neck_design_loss: 0.0381 - val_skirt_length_acc: 0.8545 - val_collar_design_acc: 0.8131 - val_coat_length_acc: 0.8499 - val_pant_length_acc: 0.8483 - val_neckline_design_acc: 0.8577 - val_lapel_design_acc: 0.8234 - val_sleeve_length_acc: 0.8697 - val_neck_design_acc: 0.7461\n",
      "Epoch 2/3\n",
      "1990/1990 [==============================] - 3040s 2s/step - loss: 0.2606 - skirt_length_loss: 0.0275 - collar_design_loss: 0.0299 - coat_length_loss: 0.0387 - pant_length_loss: 0.0200 - neckline_design_loss: 0.0625 - lapel_design_loss: 0.0173 - sleeve_length_loss: 0.0393 - neck_design_loss: 0.0254 - skirt_length_acc: 0.8938 - collar_design_acc: 0.8746 - coat_length_acc: 0.9028 - pant_length_acc: 0.8820 - neckline_design_acc: 0.9028 - lapel_design_acc: 0.8885 - sleeve_length_acc: 0.9103 - neck_design_acc: 0.7941 - val_loss: 0.4006 - val_skirt_length_loss: 0.0386 - val_collar_design_loss: 0.0443 - val_coat_length_loss: 0.0593 - val_pant_length_loss: 0.0324 - val_neckline_design_loss: 0.0994 - val_lapel_design_loss: 0.0313 - val_sleeve_length_loss: 0.0576 - val_neck_design_loss: 0.0376 - val_skirt_length_acc: 0.8546 - val_collar_design_acc: 0.8135 - val_coat_length_acc: 0.8502 - val_pant_length_acc: 0.8590 - val_neckline_design_acc: 0.8616 - val_lapel_design_acc: 0.8357 - val_sleeve_length_acc: 0.8771 - val_neck_design_acc: 0.7506\n",
      "Epoch 3/3\n",
      "1990/1990 [==============================] - 3039s 2s/step - loss: 0.2268 - skirt_length_loss: 0.0244 - collar_design_loss: 0.0258 - coat_length_loss: 0.0337 - pant_length_loss: 0.0175 - neckline_design_loss: 0.0552 - lapel_design_loss: 0.0139 - sleeve_length_loss: 0.0338 - neck_design_loss: 0.0224 - skirt_length_acc: 0.9022 - collar_design_acc: 0.8864 - coat_length_acc: 0.9156 - pant_length_acc: 0.8942 - neckline_design_acc: 0.9120 - lapel_design_acc: 0.8995 - sleeve_length_acc: 0.9294 - neck_design_acc: 0.8086 - val_loss: 0.3984 - val_skirt_length_loss: 0.0380 - val_collar_design_loss: 0.0448 - val_coat_length_loss: 0.0584 - val_pant_length_loss: 0.0310 - val_neckline_design_loss: 0.0998 - val_lapel_design_loss: 0.0322 - val_sleeve_length_loss: 0.0573 - val_neck_design_loss: 0.0367 - val_skirt_length_acc: 0.8595 - val_collar_design_acc: 0.8226 - val_coat_length_acc: 0.8503 - val_pant_length_acc: 0.8603 - val_neckline_design_acc: 0.8641 - val_lapel_design_acc: 0.8359 - val_sleeve_length_acc: 0.8773 - val_neck_design_acc: 0.7527\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f83e3921b70>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=[acc])\n",
    "model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=3, validation_data=(X_valid, y_valid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T22:58:21.988308Z",
     "start_time": "2018-04-07T22:02:58.262769Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/1\n",
      "1990/1990 [==============================] - 3275s 2s/step - loss: 0.1859 - skirt_length_loss: 0.0203 - collar_design_loss: 0.0215 - coat_length_loss: 0.0262 - pant_length_loss: 0.0149 - neckline_design_loss: 0.0456 - lapel_design_loss: 0.0118 - sleeve_length_loss: 0.0269 - neck_design_loss: 0.0186 - skirt_length_acc: 0.9141 - collar_design_acc: 0.9001 - coat_length_acc: 0.9367 - pant_length_acc: 0.9030 - neckline_design_acc: 0.9278 - lapel_design_acc: 0.9106 - sleeve_length_acc: 0.9429 - neck_design_acc: 0.8237 - val_loss: 0.3961 - val_skirt_length_loss: 0.0376 - val_collar_design_loss: 0.0453 - val_coat_length_loss: 0.0584 - val_pant_length_loss: 0.0313 - val_neckline_design_loss: 0.0993 - val_lapel_design_loss: 0.0315 - val_sleeve_length_loss: 0.0570 - val_neck_design_loss: 0.0357 - val_skirt_length_acc: 0.8550 - val_collar_design_acc: 0.8224 - val_coat_length_acc: 0.8563 - val_pant_length_acc: 0.8618 - val_neckline_design_acc: 0.8626 - val_lapel_design_acc: 0.8400 - val_sleeve_length_acc: 0.8751 - val_neck_design_acc: 0.7569\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f83179491d0>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2.compile(optimizer=Adam(1e-6), loss='categorical_crossentropy', metrics=[acc])\n",
    "model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=1, validation_data=(X_valid, y_valid))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 计算验证集准确率"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T22:58:53.275440Z",
     "start_time": "2018-04-07T22:58:21.990355Z"
    }
   },
   "outputs": [],
   "source": [
    "model.save('model_%s.h5' % model_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T23:01:37.022671Z",
     "start_time": "2018-04-07T22:58:53.276890Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "15915/15915 [==============================] - 164s 10ms/step\n",
      "skirt_length 0.8871057000553404\n",
      "collar_design 0.8586102719033233\n",
      "coat_length 0.8683015440508629\n",
      "pant_length 0.89374185136897\n",
      "neckline_design 0.8618533257532689\n",
      "lapel_design 0.8917018284106891\n",
      "sleeve_length 0.8756613756613757\n",
      "neck_design 0.8346595932802829\n"
     ]
    }
   ],
   "source": [
    "y_pred = model2.predict(X_valid, batch_size=128, verbose=1)\n",
    "a = np.array([x.any(axis=-1) for x in y_valid]).T.astype('uint8')\n",
    "b = [np.where((a == np.eye(8)[x]).all(axis=-1))[0] for x in range(8)]\n",
    "for c in range(8):\n",
    "    y_pred2 = y_pred[c][b[c]].argmax(axis=-1)\n",
    "    y_true2 = y_valid[c][b[c]].argmax(axis=-1)\n",
    "    print(label_names[c], (y_pred2 == y_true2).mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T23:01:37.038867Z",
     "start_time": "2018-04-07T23:01:37.025054Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({'coat_length': 11320,\n",
       "         'collar_design': 8393,\n",
       "         'lapel_design': 7034,\n",
       "         'neck_design': 5696,\n",
       "         'neckline_design': 17148,\n",
       "         'pant_length': 7460,\n",
       "         'skirt_length': 9223,\n",
       "         'sleeve_length': 13299})"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "counts = Counter(df.label_name)\n",
    "counts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T23:01:37.050973Z",
     "start_time": "2018-04-07T23:01:37.040700Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8713447116844438\n"
     ]
    }
   ],
   "source": [
    "s = 0\n",
    "n = 0\n",
    "for c in range(8):\n",
    "    y_pred2 = y_pred[c][b[c]].argmax(axis=-1)\n",
    "    y_true2 = y_valid[c][b[c]].argmax(axis=-1)\n",
    "    s += counts[label_names[c]] * (y_pred2 == y_true2).mean()\n",
    "    n += counts[label_names[c]]\n",
    "print(s / n)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 在测试集上预测"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T23:01:37.075633Z",
     "start_time": "2018-04-07T23:01:37.052291Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>filename</th>\n",
       "      <th>label_name</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Images/collar_design_labels/60951f5761ea3a01d6...</td>\n",
       "      <td>collar_design_labels</td>\n",
       "      <td>?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Images/collar_design_labels/677e1183282769a3fe...</td>\n",
       "      <td>collar_design_labels</td>\n",
       "      <td>?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Images/collar_design_labels/3f5a2be205e5d253eb...</td>\n",
       "      <td>collar_design_labels</td>\n",
       "      <td>?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Images/collar_design_labels/1712bf3d7b57d779ad...</td>\n",
       "      <td>collar_design_labels</td>\n",
       "      <td>?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Images/collar_design_labels/0b2b4254f35ce3a41a...</td>\n",
       "      <td>collar_design_labels</td>\n",
       "      <td>?</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                            filename            label_name  \\\n",
       "0  Images/collar_design_labels/60951f5761ea3a01d6...  collar_design_labels   \n",
       "1  Images/collar_design_labels/677e1183282769a3fe...  collar_design_labels   \n",
       "2  Images/collar_design_labels/3f5a2be205e5d253eb...  collar_design_labels   \n",
       "3  Images/collar_design_labels/1712bf3d7b57d779ad...  collar_design_labels   \n",
       "4  Images/collar_design_labels/0b2b4254f35ce3a41a...  collar_design_labels   \n",
       "\n",
       "  label  \n",
       "0     ?  \n",
       "1     ?  \n",
       "2     ?  \n",
       "3     ?  \n",
       "4     ?  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test = pd.read_csv('rank/Tests/question.csv', header=None)\n",
    "df_test.columns = ['filename', 'label_name', 'label']\n",
    "\n",
    "fnames_test = df_test.filename\n",
    "\n",
    "n_test = len(df_test)\n",
    "df_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T23:01:53.661369Z",
     "start_time": "2018-04-07T23:01:37.077016Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 10080/10080 [00:14<00:00, 682.05it/s]\n"
     ]
    }
   ],
   "source": [
    "def f(index):\n",
    "    return index, cv2.resize(cv2.imread('rank/'+fnames_test[index]), (width, width))\n",
    "\n",
    "X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)\n",
    "with multiprocessing.Pool(12) as pool:\n",
    "    with tqdm(pool.imap_unordered(f, range(n_test)), total=n_test) as pbar:\n",
    "        for i, img in pbar:\n",
    "            X_test[i] = img[:,:,::-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T23:03:12.008784Z",
     "start_time": "2018-04-07T23:01:53.664178Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10080/10080 [==============================] - 78s 8ms/step\n"
     ]
    }
   ],
   "source": [
    "y_pred = model2.predict(X_test, batch_size=128, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T23:03:13.281597Z",
     "start_time": "2018-04-07T23:03:12.009929Z"
    }
   },
   "outputs": [],
   "source": [
    "for i in range(n_test):\n",
    "    problem_name = df_test.label_name[i].replace('_labels', '')\n",
    "    problem_index = label_names.index(problem_name)\n",
    "    probs = y_pred[problem_index][i]\n",
    "    df_test.label[i] = ';'.join(np.char.mod('%.8f', probs))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T23:03:13.360156Z",
     "start_time": "2018-04-07T23:03:13.282708Z"
    }
   },
   "outputs": [],
   "source": [
    "fname_csv = 'pred_%s.csv' % model_name\n",
    "fname_zip = 'pred_%s.zip' % model_name\n",
    "df_test.to_csv(fname_csv, index=None, header=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-04-07T23:03:13.758745Z",
     "start_time": "2018-04-07T23:03:13.362310Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  adding: pred_nasnet_fine_tuning_8clf_2.csv (deflated 69%)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "rm: 无法删除'pred_nasnet_fine_tuning_8clf_2.zip': 没有那个文件或目录\n"
     ]
    }
   ],
   "source": [
    "%%bash -s $fname_csv $fname_zip\n",
    "rm $2\n",
    "zip $2 $1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.6.4"
  },
  "toc": {
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": "block",
   "toc_window_display": false
  },
  "varInspector": {
   "cols": {
    "lenName": "40",
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "position": {
    "height": "456px",
    "left": "980px",
    "right": "20px",
    "top": "134px",
    "width": "521px"
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
