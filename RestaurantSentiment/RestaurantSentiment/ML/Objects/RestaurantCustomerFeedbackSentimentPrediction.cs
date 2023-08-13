using Microsoft.ML.Data;

namespace RestaurantSentiment.ML.Objects;

// Provides the output class for our model
internal class RestaurantCustomerFeedbackSentimentPrediction : RestaurantCustomerFeedback
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    // The confidence of the models prediction
    public float Probability { get; set; }

    // Used for evaluation
    public float Score { get; set; }
}