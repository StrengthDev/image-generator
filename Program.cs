using System;
using System.IO;
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
            NDArray trainData = IO.GetTrainData(trainPath);
            Console.WriteLine("Done processing data");
        }
    }
}
