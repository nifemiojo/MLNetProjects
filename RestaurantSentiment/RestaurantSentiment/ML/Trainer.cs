using Microsoft.ML;
using RestaurantSentiment.ML.Base;
using RestaurantSentiment.ML.Objects;

namespace RestaurantSentiment.ML;

internal class Trainer : BaseML
{
    public void Train()
    {
        if (!File.Exists("data/sampledata.csv"))
        {
            Console.WriteLine($"Failed to find training data file sampledata.csv");

            return;
        }

        IDataView dataView = MlContext.Data.LoadFromTextFile<RestaurantCustomerFeedback>("data/sampledata.csv");

        // Split the data into a training and test set
        // 80% of the data will be used for training, 20% for testing
        var trainTestData = MlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

        // Adds new column to the dataset, called "Features", which contains a numeric vector
        // This will convert the text into a numeric vector
        var estimator = MlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(RestaurantCustomerFeedback.Text));
        
        // Create instance of the binary classification trainer
        var sdcaRegressionTrainer = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(RestaurantCustomerFeedback.Label),
            featureColumnName: "Features");

        var estimatorChain = estimator.Append(sdcaRegressionTrainer);

        // Create and train the model
        ITransformer trainedModel = estimatorChain.Fit(trainTestData.TrainSet);
        // Save the model to file
        MlContext.Model.Save(trainedModel, trainTestData.TrainSet.Schema, ModelPath);

        // Lazy load test data
        var testSetDataView = trainedModel.Transform(trainTestData.TestSet);

        var modelMetrics = MlContext.BinaryClassification.Evaluate(
            data: testSetDataView,
            labelColumnName: nameof(RestaurantCustomerFeedback.Label),
            scoreColumnName: nameof(RestaurantCustomerFeedbackSentimentPrediction.Score));

        Console.WriteLine($"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}{Environment.NewLine}" +
                          $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}{Environment.NewLine}" +
                          $"Accuracy: {modelMetrics.Accuracy:P2}{Environment.NewLine}" +
                          $"F1Score: {modelMetrics.F1Score:P2}{Environment.NewLine}" +
                          $"Positive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
                          $"Negative Recall: {modelMetrics.NegativeRecall}{Environment.NewLine}");
    }
}