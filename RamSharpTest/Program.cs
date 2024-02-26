using System;
using System.Collections.Generic;
using System.Drawing;

namespace RamSharpTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Bitmap bitmap = new Bitmap(@".\img\demo.png");
            string tagListPath = @".\model\ram_tag_list.txt";
            string thresholdPath = @".\model\ram_tag_list_threshold.txt";
            string modelPath = @".\model\ram_model.onnx";
            RamSharp.Lib lib = new RamSharp.Lib();
            Console.WriteLine("Loading tags...");
            List<RamSharp.Tag> tags = lib.GetModelTagList(tagListPath, thresholdPath);
            Console.WriteLine("Predicting...");
            double[] scores = lib.Predict(modelPath,bitmap);
            Console.WriteLine("Getting top tags...");
            Console.WriteLine();
            Console.WriteLine("Top tags:");
            List<RamSharp.Tag> topTags = lib.GetTopTags(scores, tags);
            foreach (var tag in topTags)
            {
                Console.WriteLine($"tag: {tag.Name} , score: {tag.Score}");
            }
            Console.WriteLine();
            Console.WriteLine("Press any key to exit");
            Console.ReadKey();
        }
    }
}
