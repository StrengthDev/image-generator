using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using static Tensorflow.Keras.Layers.LayersApi;
using Tensorflow;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Layers;

namespace image_generator
{
    class GANUtils
    {
        private const int FILTER_NUM = 128;
        private const float ALPHA = 0.3f;
        private const float BETA = 0.5f;
        private const float DROPOUT = 0.4f;
        private const float LEARNING_RATE = 0.0002f;

        public static Tensors generator(TensorShape input_shape, int inputHW, int outputHW, int channels)
        {
            LayersApi layers = new LayersApi();
            int HW = inputHW;

            Tensor inputs = keras.Input(input_shape);
            Tensors x = layers.Dense(FILTER_NUM * inputHW * inputHW).Apply(inputs);
            x = layers.LeakyReLU(ALPHA).Apply(x);
            x = layers.Reshape((inputHW, inputHW, FILTER_NUM)).Apply(x);
            x = layers.UpSampling2D().Apply(x);
            HW *= 2;
            while(!(outputHW <= HW))
            {
                x = layers.Conv2D(FILTER_NUM, (4, 4), padding: "same", activation: (string)null).Apply(x); //overloads forcing me to cast a null
                x = layers.UpSampling2D().Apply(x);
            }
            if(HW != outputHW)
            {
                throw new ArgumentException("Output height/width must obtainable through theough the expression: Input * 2 ^ N.");
            }
            x = layers.Conv2D(FILTER_NUM, (4, 4), padding: "same", activation: (string)null).Apply(x);
            x = layers.LeakyReLU(ALPHA).Apply(x);
            x = layers.Conv2D(channels, (7, 7), padding: "same", activation: (string)null).Apply(x);
            return x;
        }

        public static Tensorflow.Keras.Engine.Functional discriminator(int inputHW, int channels)
        {
            LayersApi layers = new LayersApi();
            LossesApi losses = new LossesApi();

            Tensor input = layers.Input((inputHW, inputHW, channels));
            Tensors x = layers.Conv2D(FILTER_NUM / 2, (3, 3), (2, 2), "same", activation: (string)null).Apply(input);
            x = layers.LeakyReLU(ALPHA).Apply(x);
            x = layers.Dropout(DROPOUT).Apply(x);
            x = layers.Conv2D(FILTER_NUM / 2, (3, 3), (2, 2), "same", activation: (string)null).Apply(input);
            x = layers.LeakyReLU(ALPHA).Apply(x);
            x = layers.Dropout(DROPOUT).Apply(x);

            x = layers.Flatten().Apply(x);
            x = layers.Dense(1, "sigmoid").Apply(x);
            var loss = losses.CategoricalCrossentropy();
            Adam opt = new Adam(LEARNING_RATE, BETA);
            Tensorflow.Keras.Engine.Functional model = keras.Model(input, x);
            model.compile(loss, opt, new[] {"accuracy"});
            return model;
        }

        public static Tensorflow.Keras.Engine.Functional GAN(Tensors generator, Tensorflow.Keras.Engine.Functional discriminator)
        {
            return null;
        }
    }
}
