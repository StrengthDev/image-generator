using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
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

        public static NDArray GetTrainData(string path)
        {
            bool first = true;
            NDArray data = null;
            foreach(string p in Directory.GetFiles(path))
            {
                if(p.EndsWith(".png"))
                {
                    Console.WriteLine("Processing.. " + p);
                    using(Bitmap bmp = new Bitmap(p))
                    {
                        if(first)
                        {
                            data = ImageToMatrix(bmp);
                            first = false;
                        } else
                        {
                            data = np.concatenate((data, ImageToMatrix(bmp)), 0); //TODO: make this multi-threaded
                        }
                    }
                }
            }
            return data;
        }
    }
}
