using System;
using System.Drawing;
using System.Drawing.Imaging;
using NumSharp;

namespace image_generator
{
    class IO
    {
        public static NDArray ImageToMatrix(Bitmap image)
        {
            BitmapData bmpData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, image.PixelFormat);
            NDArray matrix = new NDArray(NPTypeCode.Byte, Shape.Vector(bmpData.Stride * image.Height), fillZeros: false);
            try
            {
                unsafe
                {
                    byte* src = (byte*)bmpData.Scan0;
                    byte* dst = (byte*)matrix.Unsafe.Address;

                    Buffer.MemoryCopy(src, dst, matrix.size, matrix.size);
                    return matrix.reshape(1, image.Height, image.Width, 4);
                }
            }
            finally
            {
                image.UnlockBits(bmpData);
            }
        }

        public static Bitmap MatrixToImage(NDArray data)
        {
            int[] dims = data.shape;
            Bitmap image = new Bitmap(dims[0], dims[1], PixelFormat.Format32bppArgb);
            BitmapData bmpData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.WriteOnly, image.PixelFormat);
            NDArray tmp = new NDArray(NPTypeCode.Byte, Shape.Vector(bmpData.Stride * image.Height), fillZeros: true);
            tmp += data.flatten();
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
    }
}
