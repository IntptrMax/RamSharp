using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace RamSharp
{
    internal class CommonLib
    {
        public static Tensor<float> ExtractPixels(Image image, Size size)
        {
            Bitmap bitmap = image.Clone() as Bitmap;
            var rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, bitmap.PixelFormat);
            int bytesPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;
            var tensor = new DenseTensor<float>(new[] { 1, 3, size.Height, size.Width });
            int count = bitmapData.Stride * bitmapData.Height;
            byte[] pixelData = new byte[count];
            Marshal.Copy(bitmapData.Scan0, pixelData, 0, count);
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var std = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < bitmapData.Height; y++)
            {
                for (int x = 0; x < bitmapData.Width; x++)
                {
                    int r = pixelData[y * bitmapData.Stride + x * 3 + 2];
                    int g = pixelData[y * bitmapData.Stride + x * 3 + 1];
                    int b = pixelData[y * bitmapData.Stride + x * 3 + 0];
                    tensor[0, 0, y, x] = (r / 255.0f - mean[0]) / std[0];
                    tensor[0, 1, y, x] = (g / 255.0f - mean[1]) / std[1];
                    tensor[0, 2, y, x] = (b / 255.0f - mean[2]) / std[2];
                }
            }
            bitmap.UnlockBits(bitmapData);
            return tensor;
        }

        public static Bitmap ResizeImage(Image image, Size size)
        {
            var resizedImage = new Bitmap(size.Width, size.Height, PixelFormat.Format24bppRgb);
            using (var graphics = Graphics.FromImage(resizedImage))
            {
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.DrawImage(image, 0, 0, size.Width, size.Height);
            }
            return resizedImage;
        }

        public static List<Tag> GetModelTagList(string filePath, string thresholdFilePath)
        {
            string[] thresholdLines = File.ReadAllLines(thresholdFilePath);
            string[] tagLines = File.ReadAllLines(filePath);
            List<Tag> tags = new List<Tag>();
            for (int i = 0; i < tagLines.Length; i++)
            {
                tags.Add(new Tag { Name = tagLines[i], Threshold = double.Parse(thresholdLines[i]) });
            }
            return tags;
        }

        private DataTable OpenCSV(string filePath)
        {
            DataTable dt = new DataTable();
            FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            StreamReader sr = new StreamReader(fs, Encoding.UTF8);

            string strLine = "";
            string[] aryLine = null;
            string[] tableHead = null;

            int columnCount = 0;
            bool IsFirst = true;

            while ((strLine = sr.ReadLine()) != null)
            {

                if (IsFirst == true)
                {
                    tableHead = strLine.Split('\t');
                    IsFirst = false;
                    columnCount = tableHead.Length;

                    for (int i = 0; i < columnCount; i++)
                    {
                        DataColumn dc = new DataColumn(tableHead[i]);
                        dt.Columns.Add(dc);
                    }
                }
                else
                {
                    aryLine = strLine.Split('\t');
                    DataRow dr = dt.NewRow();
                    for (int j = 0; j < columnCount; j++)
                    {
                        dr[j] = aryLine[j];
                    }
                    dt.Rows.Add(dr);
                }
            }

            sr.Close();
            fs.Close();
            return dt;
        }

        PixelFormat[] pixelFormatArray = {
                                            PixelFormat.Format1bppIndexed
                                            ,PixelFormat.Format4bppIndexed
                                            ,PixelFormat.Format8bppIndexed
                                            ,PixelFormat.Undefined
                                            ,PixelFormat.DontCare
                                            ,PixelFormat.Format16bppArgb1555
                                            ,PixelFormat.Format16bppGrayScale
                                        };
        private bool IsPixelFormatIndexed(PixelFormat imgPixelFormat)
        {
            foreach (PixelFormat pf in pixelFormatArray)
            {
                if (imgPixelFormat == pf)
                {
                    return true;
                }
            }
            return false;
        }
    }
}
