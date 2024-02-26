using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace RamSharp
{
    public class Lib
    {
        public double[] Predict(string modelPath, Bitmap bitmap)
        {
            InferenceSession inferenceSession = new InferenceSession(modelPath);

            Size size = new Size(384, 384);
            Bitmap resized = CommonLib.ResizeImage(bitmap, size);
            Tensor<float> tensor = CommonLib.ExtractPixels(resized, size);

            var inputs = new List<NamedOnnxValue> // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor(inferenceSession.InputNames[0], tensor)
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = inferenceSession.Run(inputs); // run inference

            DenseTensor<float> output = result[0].Value as DenseTensor<float>; // get output
            float[] array = output.ToArray();
            double [] scores = new double[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                double score = 1 / (1 + Math.Exp(array[i] * -1));
                scores[i] = score;
            }
            return scores;
        }

        public List<Tag> GetModelTagList(string filePath, string thresholdFilePath)
        {
            List<Tag> tags = CommonLib.GetModelTagList(filePath, thresholdFilePath);
            return tags;
        }

        public List<Tag> GetTopTags(double[] scores, List<Tag> tags)
        {
            List<Tag> topTags = new List<Tag>();
            for (int i = 0; i < scores.Length; i++)
            {
                if (scores[i] > tags[i].Threshold)
                {
                    tags[i].Score = scores[i];
                    topTags.Add(tags[i]);
                }
            }
            return topTags.OrderByDescending(x => x.Score).ToList();
        }

    }
}
