using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML;
using TaxiFarePrediction;



namespace TaxiFarePrediction
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("----    Entering Program.cs    ----");

            // Declare Job Variables 
            string currentDir = Environment.CurrentDirectory;
            string myProjectDir = "C:\\Users\\edyar\\source\\repos\\TaxiFarePrediction";
            string _trainDataPath = Path.Combine(myProjectDir, "Data", "taxi-fare-train.csv");
            string _testDataPath = Path.Combine(myProjectDir, "Data", "taxi-fare-test.csv");
            string _modelPath = Path.Combine(myProjectDir, "Data", "Model.zip");


            MLContext mlContexto = new MLContext(seed: 0);
            var model = Train(mlContexto, _trainDataPath);
            Console.WriteLine(model);
            Evaluate(mlContexto, model);
            TestSinglePrediction(mlContexto, model);

            ITransformer Train(MLContext mlContext, string dataPath)
            {
                IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
                var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

                var modelo = pipeline.Fit(dataView);
                return modelo;
            }


            void Evaluate(MLContext mlContext, ITransformer _model)
            {
                IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
                var predictions = _model.Transform(dataView);
                var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
                Console.WriteLine();
                Console.WriteLine($"*************************************************");
                Console.WriteLine($"*       Model quality metrics evaluation         ");
                Console.WriteLine($"*------------------------------------------------");
                Console.WriteLine("\n");
                Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
                Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
                Console.WriteLine("\n");
            }

            void TestSinglePrediction(MLContext mlContext, ITransformer _model)
            {
                var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

                var taxiTripSample = new TaxiTrip()
                {
                    VendorId = "VTS",
                    RateCode = "1",
                    PassengerCount = 1,
                    TripTime = 1140,
                    TripDistance = 3.75f,
                    PaymentType = "CRD",
                    FareAmount = 0 // To predict. Actual/Observed = 15.5
                };

                var prediction = predictionFunction.Predict(taxiTripSample);

                Console.WriteLine($"**********************************************************************");
                Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
                Console.WriteLine($"**********************************************************************");
            }

            Console.WriteLine("\n");
            Console.WriteLine("----    Exiting Program.cs    ----");

            // Keeps the Console Open 
            Console.Read();
        }
    }
}
