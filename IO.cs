using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Collections.Generic;
using System.Threading.Tasks;
using NumSharp;

namespace image_generator
{
    class IO
    {
        public static NDArray ImageToMatrix(Bitmap image)
        {
            Bitmap data;
            if(image.PixelFormat != PixelFormat.Format32bppArgb)
            {
                Bitmap reformat = new Bitmap(image.Width, image.Height, PixelFormat.Format32bppArgb);
                using (Graphics g = Graphics.FromImage(reformat))
                {
                    g.DrawImage(image, new Rectangle(0, 0, image.Width, image.Height));
                }
                data = reformat;
            } else
            {
                data = image;
            }
            BitmapData bmpData = data.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            NDArray matrix = new NDArray(NPTypeCode.Byte, Shape.Vector(bmpData.Stride * image.Height), fillZeros: false);
            try
            {
                unsafe
                {
                    byte* src = (byte*)bmpData.Scan0;
                    byte* dst = (byte*)matrix.Unsafe.Address;

                    Buffer.MemoryCopy(src, dst, matrix.size, matrix.size);
                    return matrix.reshape(1, image.Height, image.Width, 4).astype(NPTypeCode.Float) / 255f;
                }
            }
            finally
            {
                data.UnlockBits(bmpData);
            }
        }

        public static Bitmap MatrixToImage(NDArray data)
        {
            int[] dims = data.shape;
            Bitmap image = new Bitmap(dims[0], dims[1], PixelFormat.Format32bppArgb);
            BitmapData bmpData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.WriteOnly, image.PixelFormat);
            NDArray tmp = new NDArray(NPTypeCode.Byte, Shape.Vector(bmpData.Stride * image.Height), fillZeros: true);
            tmp += (data.flatten() * 255f).astype(NPTypeCode.Byte);
            try
            {
                unsafe
                {
                    byte* src = (byte*)tmp.Unsafe.Address;
                    byte* dst = (byte*)bmpData.Scan0;

                    Buffer.MemoryCopy(src, dst, data.size, data.size);
                    return image;
                }
            }
            finally
            {
                image.UnlockBits(bmpData);
            }
        }

        private static void GetTrainData(string[] images, int index, Task[] threads, NDArray[] results)
        {
            int total = images.Length;
            int base_n = total / threads.Length;
            int mod = total % threads.Length;
            int start, end;
            if(index < mod)
            {
                start = index * (base_n + 1);
                end = start + base_n + 1;
            } else
            {
                start = mod * (base_n + 1) + (index - mod) * base_n;
                end = start + base_n;
            }
            NDArray data = null;
            for(int i = start; i < end; i++)
            {
                Console.WriteLine($"Thread {index} reading file({i}).. " + images[i]);
                using (Bitmap bmp = new Bitmap(images[i]))
                {
                    if (i == start)
                    {
                        data = ImageToMatrix(bmp);
                    }
                    else
                    {
                        data = np.concatenate((data, ImageToMatrix(bmp)), 0);
                    }
                }
            }
            if(index % 2 == 1)
            {
                results[index] = data;
                return;
            }
            if(index == threads.Length - 1)
            {
                results[index] = data;
                return;
            }
            for(int d = 1; index % (d * 2) == 0 && d + index < threads.Length; d *= 2)
            {
                threads[index + d].Wait();
                Console.WriteLine($"Thread {index} concatenating with {index + d}");
                data = np.concatenate((data, results[index + d]), 0);
                results[index + d] = null;
            }
            results[index] = data;
        }

        public static NDArray GetTrainData(string path, uint nthreads = 0)
        {
            List<string> images_list = new List<string>();
            foreach (string p in Directory.GetFiles(path))
            {
                if (p.EndsWith(".png"))
                {
                    images_list.Add(p);
                }
            }
            string[] images_array = images_list.ToArray();
            Task[] threads = new Task[nthreads];
            NDArray[] results = new NDArray[nthreads];
            for (int i = 1; i < nthreads; i++)
            {
                int index = i;
                threads[i] = Task.Run(() => GetTrainData(images_array, index, threads, results));
            }
            GetTrainData(images_array, 0, threads, results);
            return results[0];
        }
    }
}
