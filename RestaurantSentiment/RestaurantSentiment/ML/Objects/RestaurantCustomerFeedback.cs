using Microsoft.ML.Data;

namespace RestaurantSentiment.ML.Objects;

// Provides the input class for our model
public class RestaurantCustomerFeedback
{
    [LoadColumn(0)]
    public bool Label { get; set; } // Positive = 0, Negative = 1

    [LoadColumn(1)]
    public string Text { get; set; }
}