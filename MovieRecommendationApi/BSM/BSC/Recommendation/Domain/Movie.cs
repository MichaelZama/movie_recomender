using Newtonsoft.Json;

namespace MovieRecommendationApi.BSM.BSC.Recommendation.Domain
{
    public class Movie
    {
        [JsonProperty("title")]
        public string Title { get; set; } = "";

        [JsonProperty("year")]
        public int Year { get; set; }

        [JsonProperty("cast")]
        public List<string> Cast { get; set; } = new();

        [JsonProperty("genres")]
        public List<string> Genres { get; set; } = new();

        [JsonProperty("href")]
        public string Href { get; set; } = "";

        [JsonProperty("extract")]
        public string Extract { get; set; } = "";

        [JsonProperty("thumbnail")]
        public string Thumbnail { get; set; } = "";

        [JsonProperty("thumbnail_width")]
        public int ThumbnailWidth { get; set; }

        [JsonProperty("thumbnail_height")]
        public int ThumbnailHeight { get; set; }

        // ID generato per Qdrant (non presente nel JSON originale)
        [JsonIgnore]
        public int Id => Title.GetHashCode() & 0x7FFFFFFF; // Genera ID positivo dal titolo

        // Testo combinato per embedding
        public string GetCombinedText()
        {
            var castList = Cast.Take(5); // Prime 5 star per non rendere il testo troppo lungo
            return $"{Title} ({Year}). {Extract}. Genres: {string.Join(", ", Genres)}. Starring: {string.Join(", ", castList)}.";
        }
    }
