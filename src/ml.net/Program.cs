using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Newtonsoft.Json;
using ml.net.dao;

namespace ml.net
{
    class Program
    {
        static void Main(string[] args)
        {
            uint trainingDurationInSeconds = 60;
            if (args != null && args.Length > 0 && uint.TryParse(args[0], out trainingDurationInSeconds))
            {
                Console.WriteLine($"Overriding training time, setting for {trainingDurationInSeconds} seconds.");
            }

            var mlContext = new MLContext();
            var trainData = mlContext.Data.LoadFromTextFile(path: "../../data/optdigits-train.csv",
                                    columns : new[] 
                                    {
                                        new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                                        new TextLoader.Column(nameof(InputData.Number), DataKind.Single, 64)
                                    },
                                    hasHeader : false,
                                    separatorChar : ','
                                    );
            var testData = mlContext.Data.LoadFromTextFile(path: "../../data/optdigits-test.csv",
                                    columns: new[]
                                    {
                                        new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                                        new TextLoader.Column(nameof(InputData.Number), DataKind.Single, 64)
                                    },
                                    hasHeader: false,
                                    separatorChar: ','
                                    );

            ExperimentResult<MulticlassClassificationMetrics> experimentResult = mlContext.Auto()
                .CreateMulticlassClassificationExperiment((uint)Math.Max(trainingDurationInSeconds, 30))
                .Execute(trainData, nameof(InputData.Number));

            var heroModel = experimentResult.BestRun;

            byte[] binaryModel;
            using (var memoryStream = new MemoryStream())
            {
                mlContext.Model.Save(heroModel.Model, null, memoryStream);
                binaryModel = memoryStream.ToArray();
            }

            Console.WriteLine($"Best model binary to save for use later out of {experimentResult.RunDetails.Count()} trained models:");
            Console.WriteLine(BitConverter.ToString(binaryModel).Substring(0, 33) + "...");

            Console.WriteLine("*********************");
            Console.WriteLine("Hero Model Training Metrics:");
            Console.WriteLine(JsonConvert.SerializeObject(heroModel.ValidationMetrics, Formatting.Indented).Substring(0, 600));

            var predictions = experimentResult.BestRun.Model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, nameof(InputData.Number));
            Console.WriteLine("*********************");
            Console.WriteLine("Hero Model Evaluation Metrics:");
            Console.WriteLine(JsonConvert.SerializeObject(metrics, Formatting.Indented).Substring(0, 600));

            var newPredictionMlContext = new MLContext();
            ITransformer predictionModel;
            using (var stream = new MemoryStream(binaryModel))
            {
                predictionModel = newPredictionMlContext.Model.Load(stream, out _);
            }

            var predEngine = newPredictionMlContext.Model.CreatePredictionEngine<InputData, OutputData>(predictionModel);
            var predictionScenarios = new InputData[]{SampleMNISTData.MNIST1,SampleMNISTData.MNIST2};

            Console.WriteLine("*********************");
            foreach (var input in predictionScenarios)
            {
                var predictionResult = predEngine.Predict(input);

                Console.WriteLine($"Predicted probability:       zero:  {predictionResult.Score[0]:0.####}");
                Console.WriteLine($"                             one :  {predictionResult.Score[1]:0.####}");
                Console.WriteLine($"                             two:   {predictionResult.Score[2]:0.####}");
                Console.WriteLine($"                             three: {predictionResult.Score[3]:0.####}");
                Console.WriteLine($"                             four:  {predictionResult.Score[4]:0.####}");
                Console.WriteLine($"                             five:  {predictionResult.Score[5]:0.####}");
                Console.WriteLine($"                             six:   {predictionResult.Score[6]:0.####}");
                Console.WriteLine($"                             seven: {predictionResult.Score[7]:0.####}");
                Console.WriteLine($"                             eight: {predictionResult.Score[8]:0.####}");
                Console.WriteLine($"                             nine:  {predictionResult.Score[9]:0.####}");
                Console.WriteLine();
            }
        }
    }
}
