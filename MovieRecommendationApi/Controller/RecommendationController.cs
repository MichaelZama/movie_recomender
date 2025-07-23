using Microsoft.AspNetCore.Mvc;
using MovieRecommendationApi.BSM.BSC.Embedding.Service;
using MovieRecommendationApi.DTO;
using MovieRecommendationApi.Infrastructure.Response;

namespace MovieRecommendationApi.Controller
{
    [ApiController]
    [Route("api/[controller]")]
    [Produces("application/json")]
    public class RecommendationController : ControllerBase
    {
        private readonly ILogger<RecommendationController> _logger;
        private readonly IRecommendationService _recommendationService;

        public RecommendationController(IRecommendationService recommendationService, ILogger<RecommendationController> logger)
        {
            _recommendationService = recommendationService;
            _logger = logger;
        }

        /// <summary>
        /// ...
        /// </summary>
        [HttpPost("similar")]
        public async Task<ActionResult<CallResult>> ProcessSimiliarity([FromBody] ProcessSimiliarityDTO request)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Populate Qdrant database with the movies inside a local JSON file.
        /// </summary>
        [HttpPost("load-data-from-json")]
        public async Task<ActionResult<CallResult>> LoadDataFromJson()
        {
            throw new NotImplementedException();
        }


    }
}
