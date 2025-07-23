namespace MovieRecommendationApi.Infrastructure.Response
{
    public class CallResult
    {
        public bool Success { get; set; }
        public object? EntityID { get; set; }
        public string ErrorMessage { get; set; }
    }
}
