using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using NumSharp;
using System.Drawing;

namespace image_generator
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Training folder path:");
            string trainPath = Console.ReadLine();

            //var ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();

            NDArray img = IO.ImageToMatrix(new Bitmap(@"Src.png"));

            Bitmap n = IO.MatrixToImage(img[0]);
            n.Save("Test.png");

        }
    }
}
