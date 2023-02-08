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
            Console.WriteLine("\n Current Working Directory: \n" + currentDir);
            string myProjectDir = currentDir;
            Console.WriteLine("\n Data Directory: \n" + myProjectDir );
            Console.WriteLine();

            string _trainDataPath = Path.Combine(myProjectDir, "Data", "taxi-fare-train.csv");
            string _testDataPath = Path.Combine(myProjectDir, "Data", "taxi-fare-test.csv");
            string _modelPath = Path.Combine(myProjectDir, "Data", "Model.zip");


            MLContext mlContext = new MLContext(seed: 0);
            var model = Train(mlContext, _trainDataPath);

            Console.WriteLine();
            Console.WriteLine(model);

            Evaluate(mlContext, model);
            TestSinglePrediction(mlContext, model);

            ITransformer Train(MLContext _mlContext, string _dataPath)
            {
                Console.WriteLine( "  --  Entering Train()  --");
                IDataView _dataView = _mlContext.Data.LoadFromTextFile<TaxiTrip>(_dataPath, hasHeader: true, separatorChar: ',');
                var pipeline = _mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(_mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                .Append(_mlContext.Regression.Trainers.FastTree());

                Console.WriteLine(_dataView);
                var modelo = pipeline.Fit(_dataView);

                Console.WriteLine("  --  Exiting Train()  --");
                return modelo;
            }


            void Evaluate(MLContext _mlContext, ITransformer _model)
            {
                Console.WriteLine("  --  Entering Evaluate()  --");
                IDataView _dataView = _mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
                var _predictions = _model.Transform(_dataView);
                var _metrics = _mlContext.Regression.Evaluate(_predictions, "Label", "Score");

                Console.WriteLine();
                Console.WriteLine($"*************************************************");
                Console.WriteLine($"*       Model quality metrics evaluation         ");
                Console.WriteLine($"*------------------------------------------------");
                Console.WriteLine("\n");
                Console.WriteLine($"*       RSquared Score:      {_metrics.RSquared:0.##}");
                Console.WriteLine($"*       Root Mean Squared Error:      {_metrics.RootMeanSquaredError:#.##}");
                Console.WriteLine("\n");
                Console.WriteLine("  --  Exiting Evaluate()  --");
            }

            void TestSinglePrediction(MLContext _mlContext, ITransformer _model)
            {
                Console.WriteLine("  --  Entering TestSinglePrediction()  --");

                var _predictionFunction = _mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

                var _taxiTripSample = new TaxiTrip()
                {
                    VendorId = "VTS",
                    RateCode = "1",
                    PassengerCount = 1,
                    TripTime = 1140,
                    TripDistance = 3.75f,
                    PaymentType = "CRD",
                    FareAmount = 0 // To predict. Actual/Observed = 15.5
                };

                var _prediction = _predictionFunction.Predict(_taxiTripSample);

                Console.WriteLine($"**********************************************************************");
                Console.WriteLine($"Predicted fare: {_prediction.FareAmount:0.####}, actual fare: 15.5");
                Console.WriteLine($"**********************************************************************");

                Console.WriteLine("  --  Exiting TestSinglePrediction()  --");

            }

            Console.WriteLine("\n");
            Console.WriteLine("----    Exiting Program.cs    ----");

            // Keeps the Console Open 
            Console.Read();
        }
    }
}
