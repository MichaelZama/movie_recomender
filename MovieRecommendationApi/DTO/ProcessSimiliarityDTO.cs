namespace MovieRecommendationApi.DTO
{
    public class ProcessSimiliarityDTO
    {
        public string Text { get; set; }
        public int Limit { get; set; } = 10;
    }
}
