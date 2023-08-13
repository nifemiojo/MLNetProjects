using Microsoft.ML;
using RestaurantSentiment.ML.Base;
using RestaurantSentiment.ML.Objects;
using System.Reflection;

namespace RestaurantSentiment.ML;

internal class Predictor : BaseML
{
    public void Predict(string inputData)
    {
        if (!File.Exists(ModelPath))
        {
            Console.WriteLine($"Failed to find model at {ModelPath}");

            return;
        }

        ITransformer mlModel;

        using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            mlModel = MlContext.Model.Load(stream, out _);
        }

        if (mlModel == null)
        {
            Console.WriteLine("Failed to load model");

            return;
        }

        // Allows you to perform a prediction on a single instance of data
        // Only use in single-threaded environment, in prod use PredictionEnginePool
        var predictionEngine = MlContext.Model.CreatePredictionEngine<RestaurantCustomerFeedback, RestaurantCustomerFeedbackSentimentPrediction>(mlModel);

        // Makes a prediction on a single row of data
        var prediction = predictionEngine.Predict(new RestaurantCustomerFeedback { Text = inputData });

        Console.WriteLine($"Based on \"{inputData}\", the feedback is predicted to be:{Environment.NewLine}{(prediction.Prediction ? "Negative" : "Positive")} at a {prediction.Probability:P0} confidence");
    }

    public void PredictBatch()
    {
        if (!File.Exists(ModelPath))
        {
            Console.WriteLine($"Failed to find model at {ModelPath}");
            return;
        }

        ITransformer mlModel;

        using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            mlModel = MlContext.Model.Load(stream, out _);
        }

        if (mlModel == null)
        {
            Console.WriteLine("Failed to load model");
            return;
        }

        IEnumerable<RestaurantCustomerFeedback> restaurantCustomerFeedbacks = new[]
        {
            new RestaurantCustomerFeedback
            {
                Text = "This was a horrible meal"
            },
            new RestaurantCustomerFeedback
            {
                Text = "I love this spaghetti."
            }
        };

        IDataView batchComments = MlContext.Data.LoadFromEnumerable(restaurantCustomerFeedbacks);

        // Lazily loaded predictions
        IDataView predictions = mlModel.Transform(batchComments);

        // Use model to predict whether comment data is Positive (0) or Negative (1).
        IEnumerable<RestaurantCustomerFeedbackSentimentPrediction> predictedResults = MlContext.Data.CreateEnumerable<RestaurantCustomerFeedbackSentimentPrediction>(predictions, reuseRowObject: false);

        Console.WriteLine();

        Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
        foreach (RestaurantCustomerFeedbackSentimentPrediction prediction in predictedResults)
        {
            Console.WriteLine($"Sentiment: {prediction.Text} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
        }
        Console.WriteLine("=============== End of predictions ===============");
    }
}