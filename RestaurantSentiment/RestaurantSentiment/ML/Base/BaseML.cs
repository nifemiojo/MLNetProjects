using Microsoft.ML;
using RestaurantSentiment.Common;

namespace RestaurantSentiment.ML.Base;

public class BaseML
{
    protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, Constants.ModelFilename);

    protected readonly MLContext MlContext;

    protected BaseML()
    {
        MlContext = new MLContext(2020);
    }
}