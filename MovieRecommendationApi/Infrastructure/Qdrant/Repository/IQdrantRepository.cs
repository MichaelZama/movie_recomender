using MovieRecommendationApi.BSM.BSC.Recommendation.Domain;

namespace MovieRecommendationApi.Infrastructure.Qdrant.Repository
{
    public interface IQdrantRepository
    {
        Task LoadMoviesAsync(List<Movie> movies);
        Task InitializeCollectionAsync(int vectorSize);
        Task<long> GetMovieCountAsync();
    }
}
