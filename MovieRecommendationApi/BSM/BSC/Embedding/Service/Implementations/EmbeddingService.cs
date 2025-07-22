using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace MovieRecommendationApi.BSM.BSC.Embedding.Service.Implementations
{
    public class EmbeddingService : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly Dictionary<string, int> _vocabulary;
        private readonly int _maxLength;

        public EmbeddingService(string modelPath, string tokenizerPath, int maxLength = 128)
        {
            // Configurazione ottimizzata per performance
            var sessionOptions = new SessionOptions
            {
                InterOpNumThreads = Environment.ProcessorCount, // Usa tutti i core
                IntraOpNumThreads = Environment.ProcessorCount,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            _session = new InferenceSession(modelPath, sessionOptions);
            _vocabulary = LoadTokenizer(tokenizerPath);
            _maxLength = maxLength;

            Console.WriteLine($"✅ Modello ONNX ottimizzato caricato con {Environment.ProcessorCount} thread");
        }

        public async Task<List<float[]>> GetBatchEmbeddingsAsync(List<string> texts, int batchSize = 32)
        {
            var results = new List<float[]>();

            //Console.WriteLine($"📊 Processando {texts.Count} embeddings in batch di {batchSize}...");

            for (int i = 0; i < texts.Count; i += batchSize)
            {
                var batch = texts.Skip(i).Take(batchSize).ToList();
                //Console.WriteLine($"   Batch {(i / batchSize) + 1}/{(texts.Count + batchSize - 1) / batchSize}: processing {batch.Count} items...");

                var batchResults = await ProcessBatchAsync(batch);
                results.AddRange(batchResults);

                // Piccola pausa per non sovraccaricare
                await Task.Delay(50);
            }

            return results;
        }

        private async Task<List<float[]>> ProcessBatchAsync(List<string> texts)
        {
            var tasks = texts.Select(text => GetEmbeddingAsync(text)).ToArray();
            var embeddings = await Task.WhenAll(tasks);
            return embeddings.ToList();
        }

        private Dictionary<string, int> LoadTokenizer(string tokenizerPath)
        {
            try
            {
                var json = File.ReadAllText(tokenizerPath);
                var tokenizer = JsonConvert.DeserializeObject<JObject>(json);

                var vocab = new Dictionary<string, int>();

                // Carica vocab principale
                if (tokenizer?["model"]?["vocab"] != null)
                {
                    var vocabObj = tokenizer["model"]["vocab"] as JObject;
                    foreach (var item in vocabObj!)
                    {
                        vocab[item.Key] = item.Value!.Value<int>();
                    }
                }

                Console.WriteLine($"📖 Loaded {vocab.Count} tokens from MPNet tokenizer");

                // ✅ AGGIORNA: Assicurati che ci siano i token MPNet (non BERT!)
                if (!vocab.ContainsKey("<pad>")) vocab["<pad>"] = 1;   // Era [PAD] = 0
                if (!vocab.ContainsKey("<unk>")) vocab["<unk>"] = 3;   // Era [UNK] = 1
                if (!vocab.ContainsKey("<s>")) vocab["<s>"] = 0;       // Era [CLS] = 2  
                if (!vocab.ContainsKey("</s>")) vocab["</s>"] = 2;     // Era [SEP] = 3

                return vocab;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️  Errore tokenizer: {ex.Message}");
                return CreateMPNetVocabulary(); // ← Nuovo fallback
            }
        }

        private object GetNestedValue(JObject obj, string path)
        {
            var parts = path.Split('.');
            JToken current = obj;

            foreach (var part in parts)
            {
                current = current?[part];
                if (current == null) return null;
            }

            return current;
        }

        private Dictionary<string, int> CreateSimpleVocabulary()
        {
            // Vocabulary di base per testing
            var vocab = new Dictionary<string, int>
            {
                ["[PAD]"] = 0,
                ["[UNK]"] = 1,
                ["[CLS]"] = 2,
                ["[SEP]"] = 3
            };

            // Aggiungi caratteri comuni
            string chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?-";
            for (int i = 0; i < chars.Length; i++)
            {
                vocab[chars[i].ToString()] = i + 4;
            }

            return vocab;
        }

        public async Task<float[]> GetEmbeddingAsync(string text)
        {
            try
            {
                var tokens = TokenizeText(text.ToLower());
                var inputIds = tokens.Select(t => (long)t).ToArray();
                var attentionMask = Enumerable.Repeat(1L, inputIds.Length).ToArray();

                // Padding/truncation
                if (inputIds.Length < _maxLength)
                {
                    var padded = new long[_maxLength];
                    var maskPadded = new long[_maxLength];

                    Array.Copy(inputIds, padded, inputIds.Length);
                    Array.Copy(attentionMask, maskPadded, attentionMask.Length);

                    // ✅ USA PAD_ID = 1, NON 0!
                    for (int i = inputIds.Length; i < _maxLength; i++)
                    {
                        padded[i] = 1; // <pad> token ID
                        maskPadded[i] = 0; // attention mask = 0 per padding
                    }

                    inputIds = padded;
                    attentionMask = maskPadded;
                }
                else
                {
                    inputIds = inputIds.Take(_maxLength).ToArray();
                    attentionMask = attentionMask.Take(_maxLength).ToArray();
                }

                // Rest of the code stays the same...
                var inputIdsTensor = new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length });
                var attentionMaskTensor = new DenseTensor<long>(attentionMask, new[] { 1, attentionMask.Length });

                var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
            // NO token_type_ids per MPNet ✅
        };

                using var results = _session.Run(inputs);
                var embeddings = results.FirstOrDefault()?.AsTensor<float>();

                if (embeddings != null)
                {
                    return MeanPooling(embeddings, attentionMask);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️  Errore durante embedding: {ex.Message}");
            }

            return GenerateRandomEmbedding(768);
        }

        private int[] TokenizeText(string text)
        {
            //Console.WriteLine($"🔍 Tokenizing: '{text}'");

            // ✅ USA TOKEN MPNET, NON BERT!
            var tokens = new List<int> { _vocabulary.GetValueOrDefault("<s>", 0) }; // Era [CLS]

            var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            foreach (var word in words.Take(126)) // Lascia spazio per <s> e </s>
            {
                var lowerWord = word.ToLower();
                //Console.WriteLine($"   Word: '{lowerWord}'");

                if (_vocabulary.ContainsKey(lowerWord))
                {
                    //Console.WriteLine($"      → Found: {_vocabulary[lowerWord]}");
                    tokens.Add(_vocabulary[lowerWord]);
                }
                else
                {
                    //Console.WriteLine($"      → Using <unk>");
                    tokens.Add(_vocabulary.GetValueOrDefault("<unk>", 3)); // Era [UNK]
                }
            }

            tokens.Add(_vocabulary.GetValueOrDefault("</s>", 2)); // Era [SEP]
            return tokens.ToArray();
        }

        private float[] MeanPooling(Tensor<float> embeddings, long[] attentionMask)
        {
            var batchSize = embeddings.Dimensions[0];
            var seqLength = embeddings.Dimensions[1];
            var hiddenSize = embeddings.Dimensions[2];

            var result = new float[hiddenSize];
            var validTokens = 0;

            for (int i = 0; i < seqLength; i++)
            {
                if (attentionMask[i] == 1)
                {
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        result[j] += embeddings[0, i, j];
                    }
                    validTokens++;
                }
            }

            // Media
            for (int i = 0; i < hiddenSize; i++)
            {
                result[i] /= Math.Max(validTokens, 1);
            }

            // Normalizzazione L2
            return NormalizeVector(result);
        }

        private float[] GenerateRandomEmbedding(int dimension)
        {
            var random = new Random();
            var embedding = new float[dimension];

            for (int i = 0; i < dimension; i++)
            {
                embedding[i] = (float)(random.NextDouble() * 2 - 1); // [-1, 1]
            }

            return NormalizeVector(embedding);
        }

        private float[] NormalizeVector(float[] vector)
        {
            var norm = Math.Sqrt(vector.Sum(x => x * x));
            if (norm > 0)
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    vector[i] /= (float)norm;
                }
            }
            return vector;
        }

        private Dictionary<string, int> CreateMPNetVocabulary()
        {
            var vocab = new Dictionary<string, int>
            {
                ["<s>"] = 0,        // Start token
                ["<pad>"] = 1,      // Padding  
                ["</s>"] = 2,       // End token
                ["<unk>"] = 3       // Unknown
            };

            // Aggiungi caratteri base
            string chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?-";
            for (int i = 0; i < chars.Length; i++)
            {
                vocab[chars[i].ToString()] = i + 4;
            }

            return vocab;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
