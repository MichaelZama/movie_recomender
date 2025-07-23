using MovieRecommendationApi.Infrastructure.Qdrant.Repository;

namespace MovieRecommendationApi.BSM.BSC.Embedding.Repository.Implementations
{
    public class RecommendationRepository :IRecommendationRepository
    {
        private readonly IQdrantRepository _qdrantRepository;
        public RecommendationRepository(IQdrantRepository qdrantRepository)
        {
            _qdrantRepository = qdrantRepository;
        }


    }
}
