using MovieRecommendationApi.BSM.BSC.Recommendation.Domain;

namespace MovieRecommendationApi.Infrastructure.Qdrant.Repository.Implementation
{
    public class QdrantRepository : IQdrantRepository
    {

        public QdrantRepository() { }

        public Task<long> GetMovieCountAsync()
        {
            throw new NotImplementedException();
        }

        public Task InitializeCollectionAsync(int vectorSize)
        {
            throw new NotImplementedException();
        }

        public Task LoadMoviesAsync(List<Movie> movies)
        {
            throw new NotImplementedException();
        }
    }
}
