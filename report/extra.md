## (7) Agreement between CLAP and MQAScore

Spearman ρ between per-clip LAION-CLAP and per-clip MQAScore (and its parts), 100 clips per row:

| audio                                        |   ρ(mqascore) |   ρ(global) |   ρ(genre) |   ρ(instrument) |   ρ(mood) |
|:---------------------------------------------|--------------:|------------:|-----------:|----------------:|----------:|
| MusicCaps reference (CLAP = retrieval score) |         0.134 |       0.219 |      0.163 |          -0.061 |     0.158 |
| cfg4                                         |         0.192 |      -0.029 |      0.152 |           0.237 |     0.054 |
| cfg7                                         |         0.132 |       0.002 |      0.083 |           0.286 |     0.084 |
| cfg10                                        |         0.085 |      -0.053 |     -0.012 |           0.304 |     0.036 |
| cfg13                                        |         0.170 |       0.084 |     -0.012 |           0.406 |     0.048 |
| apg10                                        |         0.109 |      -0.094 |      0.067 |           0.256 |     0.013 |
| apg15                                        |         0.155 |       0.065 |      0.057 |           0.333 |     0.071 |
| interval10                                   |         0.122 |      -0.002 |      0.021 |           0.319 |     0.082 |
| compose7_3                                   |         0.098 |       0.044 |      0.002 |           0.254 |     0.065 |
| pe_cfg7                                      |         0.154 |       0.292 |      0.005 |           0.246 |     0.191 |
| neg_cfg7                                     |         0.153 |       0.018 |      0.220 |           0.258 |     0.077 |
| cfg10_50steps                                |         0.154 |       0.004 |     -0.008 |           0.348 |     0.073 |
| cfg7_seed1                                   |         0.081 |       0.104 |      0.174 |           0.208 |    -0.099 |
| cfg10_seed1                                  |         0.009 |       0.030 |      0.127 |           0.195 |    -0.032 |
| interval10_seed1                             |         0.054 |       0.042 |      0.135 |           0.134 |    -0.024 |
| compose7_3_seed1                             |        -0.085 |      -0.052 |      0.049 |           0.239 |    -0.094 |

* pooled over all 15 generated runs (1500 clips): ρ = 0.114 (95% bootstrap CI [0.061, 0.167])
* caption difficulty (per-caption mean over the 15 runs): ρ(CLAP, MQAScore) = 0.137
* within a caption, across the 15 runs: mean ρ = 0.028 (do both metrics prefer the same run?)
* retrieval score vs mean generated CLAP of the same caption: ρ = -0.157
* system level (15 runs): Spearman ρ = 0.243 (p = 0.383), Kendall τ = 0.143 (p = 0.495); without neg_cfg7: ρ = 0.068 (p = 0.817)

## (8) Retrieval failure cases

Top-1 CLAP score: 10th pct 0.402, median 0.460, 90th pct 0.513. Retrieval score vs MQAScore of the retrieved clip (judge asks about the *test* caption's concepts): ρ = 0.134 (p = 0.185).

### Six lowest-scoring retrievals

|   id |   score |   MQA(ref) | test caption                                                                                                                                          | retrieved MusicCaps caption                                                                                                                            | concepts (genre / instr. / mood)                                         |
|-----:|--------:|-----------:|:------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------|
|   62 |   0.355 |       0.36 | A breakbeat-driven piece centered on a strong, rhythmic beat, unfolding with a dramatic mood that builds tension and emotional intensity.             | This song contains a evolving synth pad sound in the higher register with noise. A piano is playing a sad minor chord progression in the mid range. …  | breakbeat / drums / dramatic                                             |
|   64 |   0.364 |       0.59 | A party-loving rocknroll piece centered on piano, full of upbeat energy and lively rhythms, capturing the fun and festive spirit of a joyful …        | This is a piece from a musical. The piano is playing a repetitive tune. There is a male vocalist singing melodically. There is a playful atmosphere …  | rock and roll / piano / upbeat;party                                     |
|   45 |   0.365 |       0.8  | A fantasy soundtrack featuring an orchestra, evoking a magical and imaginative atmosphere through rich, sweeping melodies and orchestral textures.    | The low quality recording features a classical song that consists of widely spread sustained strings melody and woodwind melody. It sounds vintage, …  | soundtrack / orchestra / magical                                         |
|   58 |   0.368 |       0.74 | A driving soundtrack featuring accordion, capturing an energetic and forward-moving theme with rhythmic intensity and dynamic propulsion.             | The low quality recording features a passionate accordion melody playing over a shimmering tambourine. It sounds joyful, happy and the recording is …  | soundtrack / accordion / energetic;driving                               |
|    6 |   0.377 |       0.52 | A chaotic noise soundscape unfolds through the warm, shimmering Rhodes, layered with deep sub-bass and textured field recordings, creating a dense, … | The low quality recording features a creepy music box melody followed by high pitched buzz, wide distorted low tone sounds and some growling noises. … | noise / Rhodes electric piano;sub-bass;field recordings / chaotic;uneasy |
|   41 |   0.389 |       0.99 | A psychedelic soundscape crafted with the warm, resonant Rhodes, evoking a dreamy, otherworldly atmosphere.                                           | This is a chiptune piece. The only instrument being used in this track is a granular synth playing the melody at a medium-to-high pitch. It is an …    | psychedelic / Rhodes electric piano / dreamy                             |

### Five references the MQAScore judge rates lowest

|   id |   retr. score |   MQA(ref) | genre / instr. / mood   | test concepts                   | retrieved aspects                                                                                               |
|-----:|--------------:|-----------:|:------------------------|:--------------------------------|:----------------------------------------------------------------------------------------------------------------|
|   65 |         0.398 |       0.28 | 0.08 / 0.73 / 0.02      | noise / Hammond organ / epic    | organ, operatic organ playing, haunting, live recording, poor audio quality, reverb, room acoustics             |
|   90 |         0.448 |       0.34 | 0.00 / 0.06 / 0.96      | chanson / bass / nostalgic      | steelpan, afrobeat, upbeat, chill, smooth, bright, positive, no vocals, fast tempo, danceable, percussion synth |
|   78 |         0.42  |       0.36 | 0.13 / 0.42 / 0.52      | rock / viola / sci-fi           | cinematic, experimental, e-bass, acoustic guitars, string section, digital acoustic drums, uptempo, complex     |
|   62 |         0.355 |       0.36 | 0.03 / 0.09 / 0.97      | breakbeat / drums / dramatic    | pop, synthesizer evolving pad sound, piano, digital drums, very sad atmosphere, medium to uptempo               |
|   57 |         0.477 |       0.39 | 0.00 / 0.51 / 0.65      | classical / orchestra / melodic | brass performance, trumpets, bass horns, slow tempo                                                             |

### Hub clips: 11 MusicCaps clips are top-1 for more than one caption (42 captions in total; 69 distinct clips for 100 captions)

| ytid        |   times retrieved | test ids                                       |   mean score | MusicCaps caption                                                                                                       |
|:------------|------------------:|:-----------------------------------------------|-------------:|:------------------------------------------------------------------------------------------------------------------------|
| 7k3OZ_fPXuM |                12 | 12, 19, 20, 26, 33, 51, 79, 84, 86, 87, 97, 98 |        0.478 | A voice sample is ringing out with a lot of reverb. A synth pad is underlining an acoustic piano being played with a …  |
| hLn4HokDEmM |                 9 | 2, 8, 15, 16, 21, 38, 80, 82, 96               |        0.506 | Audio from two songs of EDM music featuring four on the floor kick pattern, synthesized pan flute, syncopated synth …   |
| aIIEI33EUqI |                 4 | 1, 27, 50, 95                                  |        0.493 | This song contains soft synth pads playing in the lower to higher register with little dreamy notes sprinkled on top. … |
| kmZFn-CQ19A |                 3 | 13, 23, 48                                     |        0.451 | This audio contains someone playing an acoustic guitar fingerstyle. An upright bass is playing a line in the …          |
| C8Euv69GR3U |                 2 | 7, 18                                          |        0.456 | The Rock song features a wide, mellow, high pitched synth pad swelling into the drop that consists of punchy drums, …   |

### Mean retrieval score by test genre (genres with ≥ 2 captions)

| genre             |   mean |   count |
|:------------------|-------:|--------:|
| breakbeat         |  0.373 |       2 |
| noise             |  0.388 |       2 |
| soundtrack        |  0.396 |       3 |
| singer-songwriter |  0.424 |       2 |
| folk rock         |  0.426 |       2 |
| pop folk          |  0.438 |       3 |
| rock and roll     |  0.441 |       5 |
| film score        |  0.447 |       3 |
| rock              |  0.448 |       3 |
| psychedelic       |  0.449 |       3 |
| dance             |  0.452 |       2 |
| world fusion      |  0.455 |       3 |
| ambient           |  0.458 |       6 |
| hip hop           |  0.458 |       3 |
| thrash metal      |  0.458 |       2 |
| hard rock         |  0.459 |       3 |
| fusion            |  0.463 |       2 |
| electronic        |  0.464 |       4 |
| chanson           |  0.466 |       2 |
| classical         |  0.467 |       2 |
| latin             |  0.470 |       3 |
| folk              |  0.482 |       2 |
| experimental      |  0.484 |       2 |
| electronica       |  0.488 |       2 |
| house             |  0.498 |       5 |
| orchestral        |  0.505 |       4 |
| techno            |  0.526 |       3 |
| club music        |  0.537 |       2 |
