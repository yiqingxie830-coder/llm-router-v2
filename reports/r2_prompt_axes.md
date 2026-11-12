# The two axes of the R2 model-vector plot: projecting prompts onto them

**In one sentence:** PC1 is not "hard ↔ easy" but "prompts where the strong-weak gap opens up ↔ prompts where it does not". The low end is open-ended advice and recommendations; the high end is code, logic puzzles, and questions with format constraints and a definite answer. PC2 is the one related to difficulty, but it is also a language axis: the low end is short English common-knowledge questions, the high end long professional tasks in Chinese and Japanese. PC1 correlates −0.76 with the rating because the strong-weak gap really varies with the prompt, not because the rating-gap term failed to absorb average strength.

## 1. Method
- **Final model:** R2 trained on the whole dev set (d = 128, learning rate 1e-3, λ = 1e-5, 14 epochs); parameters saved in `data/r2_final_params.npz`, identical to the earlier exported vectors.
- **Prompt coordinates on the axes:** model m's "prompt-dependent fit" on prompt q equals the inner product of the model's whitened vector and the prompt's whitened vector. Projecting the prompt's whitened vector onto the same PC directions as the plot gives t1 and t2. Over all 85,835 dev-set prompts, t1 and t2 have mean 0 and variance 1 and are uncorrelated (correlation 0.01).
- **How to read the direction:** a prompt with high t1 favours models with a high PC1 coordinate. In the plot the strong tier sits at the negative end of PC1, so **the lower t1, the bigger the strong tier's edge; the higher t1, the smaller the strong-weak gap**. PC2 likewise: high-t2 prompts favour the Claude 4 series and gemini-2.5-pro, low-t2 prompts favour llama-3.3, nova-pro and gemma.
- **Task types:** the LLM tags shipped with the raw Arena data (category_tag v0.1): code, math, creative writing, instruction following, and 7 "difficulty criteria" (complexity, creativity, domain knowledge, problem solving, real world, specificity, technical accuracy). Following Arena, meeting ≥ 6 criteria counts as "hard". These tags were produced by Arena with an LLM, not by human annotators.
- **Out-of-fold check:** in-sample scoring uses votes the model has seen and may be optimistic. So each of the 5 folds trains a model with the same settings on its train votes only, runs its own PCA, and scores t1 for its own eval votes. The correlation between each fold's PC1 model coordinates and the final model's PC1 is between 0.985 and 0.993, so the axis is very stable.
- Dev set only; the test set did not take part.

## 2. Naming the two axes (decided 2026-11-10: PC1 = open-ended advice ↔ code and definite-answer questions; PC2 = short common knowledge ↔ hard professional tasks, heavily overlapping with language)

| Axis | Low end | High end | Name proposed at the time |
|---|---|---|---|
| PC1 (65%) | Open-ended advice, recommendations, learning paths ("how can I make money with AI", "recommend 3 books", "what to bring back from Turkey"). Short, multilingual, 30% Polish | Pasted code to fix, small programs, logic paradoxes, strict format constraints ("answer in one word", "put the answer in \boxed{}"), game-rule puzzles. 85% English, 53% code, many long code pastes | **Open-ended advice ↔ code and definite-answer questions** (or "how far the strong-weak gap opens") |
| PC2 (16%) | One-line English common-knowledge questions ("who is the prime minister of Singapore", "is the sky blue"). 97% English, median length 30 characters, 2.5% hard | Long professional tasks in Chinese and Japanese (HLS accelerators, motor-control code, physics problems). 66% Chinese, 10% Japanese, 31% math, 75% hard | **Short English common knowledge ↔ hard professional tasks in Chinese** |

## 3. PC1 correlates −0.76 with the rating: why?

The answer is that **the strong-weak gap varies with the prompt**, not that the rating-gap term left some average strength unabsorbed. Three pieces of evidence.

**1. By construction, PC1 cannot carry average strength.** The mean vector over all prompts is subtracted before projection, so t1 averages 0 over all prompts. That means PC1's average contribution to every model is 0. If the rating-gap term had not absorbed all of the average strength, the remainder would land in each model's bias u. In fact u correlates 0.91 with the rating, so a little does remain. But u only ranges from −0.07 to +0.05, roughly ±10 rating points, so nearly all average strength has been absorbed by the rating-gap term.

**2. In the data, the strong tier's win rate varies a lot with t1.** If this were leftover average strength, the strong tier would win at about the same rate in every quintile. What actually happens (strong-vs-weak votes, out-of-fold scores):

| t1 quintile | 1 (lowest) | 2 | 3 | 4 | 5 (highest) |
|---|---|---|---|---|---|
| Strong win % | 75.0 | 67.9 | 60.0 | 55.9 | 50.4 |
| Tie % | 13.0 | 16.8 | 22.3 | 23.8 | 28.1 |
| Weak win % | 12.0 | 15.2 | 17.6 | 20.4 | 21.5 |
| All votes: quality per 100 rating points | 0.17 | 0.14 | 0.11 | 0.10 | 0.08 |
| All votes: mean rating gap | 73.3 | 72.2 | 71.8 | 72.1 | 71.2 |
| Hard % | 32.9 | 36.2 | 36.2 | 38.7 | 42.8 |

- The strong tier's win rate falls from 75% to 50%, and the effect of the rating gap halves. The mean rating gap is between 71 and 73 points in every quintile, so the change is not due to a different mix of pairs. The in-sample version falls from 77% to 49%, almost the same as out-of-fold, so it is not overfitting either.
- **The two ends of PC1 are not "hard" and "easy".** By Arena's hard tag the five quintiles range only from 33% to 43%, and the end where the strong tier's edge is smaller is slightly harder.

**3. An intuitive explanation (a conjecture, not checked case by case):** open-ended advice has no reference answer, so votes depend on the breadth, structure and richness of the answer, which amplifies the strong models' edge. Code, logic and format-constrained questions have a clear right answer that the weak model often also gets, or both get wrong, so there are more ties and smaller gaps. PC1 correlates with the rating because, on the questions where the gap opens up, the extra edge lines up with strength. That is what "the strong-weak gap varies with the prompt" means.

**What this means for routing:** open-ended advice is worth sending to the strong model; code and definite-answer questions lose less when sent to the cheap one. Also, the weak-tier Claude 3.5 / 3.7 Sonnet sit relatively well at the high end of PC1, which may mean Arena's overall rating underrates their coding. This is only a conjecture.

## 4. More on PC2
- PC2's rank correlation with the number of difficulty criteria is 0.40 (PC1 only 0.04), so the real "difficulty axis" is PC2. But in this data most hard questions are professional questions in Chinese and Japanese, so difficulty and language cannot be separated.
- The strong tier's win rate rises only from 57% to 65% along t2, a much smaller effect than PC1. PC2 mainly distinguishes which weaker models lose out: llama-3.3, nova-pro, gemma and similar models do relatively worse on professional Chinese questions and not on short English common-knowledge questions.

## 5. Files
- `data/r2_prompt_axes.parquet`: t1, t2, t3, out-of-fold t1, language, length and tags for every dev-set prompt
- `reports/r2_prompt_axes_ends.csv`: 200 prompts from each of the four ends, with text
- `scripts/prompt_axes.py` (projection and out-of-fold check), `scripts/prompt_axes_report.py` (tables and samples)

---

Below are the full tables and raw prompt samples generated by the script. The samples are quoted from the dataset (CC-BY-4.0) in their original languages.

## A. Profile of 200 prompts at each of the four ends (85,835 distinct dev-set prompts)

|                               |   PC1 low end |   PC1 high end |   PC2 low end |   PC2 high end |   whole dev set |
|:------------------------------|--------------:|---------------:|--------------:|---------------:|----------------:|
| code %                        |          28.5 |           52.5 |           3   |           44   |            29   |
| math %                        |           0.5 |           13.5 |           1.5 |           30.5 |             8   |
| creative writing %            |           1   |           16.5 |           3   |            7   |             8.1 |
| instruction following %       |           4   |           18.5 |           0.5 |           38.5 |            16.6 |
| difficulty criteria met (0–7) |           4.5 |            4.5 |           1.7 |            6.2 |             4.4 |
| hard (≥6 criteria) %          |          26.5 |           40.5 |           2.5 |           75   |            37.8 |
| length P25 (chars)            |          45.8 |           41   |          23   |           59.8 |            45   |
| length median                 |          76   |          108   |          29.5 |          105   |            88   |
| length P75                    |         113.2 |         1205.8 |          38   |          200   |           212   |
| English %                     |          44.5 |           84.5 |          97   |            3   |            55.7 |
| Polish %                      |          29.5 |            0   |           0   |            0   |            10.7 |
| Chinese (simpl.+trad.) %      |           6.5 |            0   |           0   |           66   |             4.2 |
| Japanese %                    |           3   |            0   |           0   |           10   |             1.5 |
| und (undetermined) %          |           1.5 |           12   |           2.5 |           15.5 |             5.9 |

Top 6 languages per end: PC1 low end: en 89, pl 59, zh 13, ru 10, ko 8, ja 6; PC1 high end: en 169, und 24, ru 3, ro 1, de 1, cs 1; PC2 low end: en 194, und 5, ko 1; PC2 high end: zh 127, und 31, ja 20, ru 8, en 6, zh-Hant 5

## B. By t1 quintile (in-sample: the final model scores all dev-set votes)

|             |   votes |   higher-rated win % |   tie % |   loss % |   rating-gap slope (per 100 pts) |   mean rating gap |   strong-vs-weak votes |   strong win % |   strong-vs-weak tie % |   weak win % |   hard % |   length median |
|:------------|--------:|---------------------:|--------:|---------:|---------------------------------:|------------------:|-----------------------:|---------------:|-----------------------:|-------------:|---------:|----------------:|
| 1 (lowest)  |   18185 |                54.32 |   22.27 |    23.41 |                             0.18 |             73.17 |                   1744 |          76.72 |                  12.67 |        10.61 |    34.03 |              77 |
| 2           |   18184 |                50.06 |   25.26 |    24.68 |                             0.14 |             72.3  |                   1706 |          68.82 |                  16.53 |        14.65 |    36.38 |              81 |
| 3           |   18185 |                47.09 |   27.1  |    25.8  |                             0.12 |             71.81 |                   1692 |          61.94 |                  21.51 |        16.55 |    36.87 |              86 |
| 4           |   18184 |                43.33 |   29.09 |    27.58 |                             0.08 |             71.93 |                   1584 |          52.4  |                  25.38 |        22.22 |    38.02 |              96 |
| 5 (highest) |   18185 |                39.16 |   32.05 |    28.78 |                             0.08 |             71.07 |                   1607 |          48.91 |                  28.06 |        23.02 |    41.45 |             130 |

## C. By t1 quintile (out-of-fold: each fold's model scores only eval votes it never saw)

|             |   votes |   higher-rated win % |   tie % |   loss % |   rating-gap slope (per 100 pts) |   mean rating gap |   strong-vs-weak votes |   strong win % |   strong-vs-weak tie % |   weak win % |   hard % |   length median |
|:------------|--------:|---------------------:|--------:|---------:|---------------------------------:|------------------:|-----------------------:|---------------:|-----------------------:|-------------:|---------:|----------------:|
| 1 (lowest)  |   18185 |                53.45 |   22.59 |    23.96 |                             0.17 |             73.34 |                   1743 |          74.99 |                  13.02 |        11.99 |    32.86 |              76 |
| 2           |   18184 |                49.11 |   25.54 |    25.35 |                             0.14 |             72.18 |                   1706 |          67.94 |                  16.82 |        15.24 |    36.21 |              81 |
| 3           |   18185 |                47.15 |   26.79 |    26.07 |                             0.11 |             71.78 |                   1683 |          60.01 |                  22.34 |        17.65 |    36.22 |              84 |
| 4           |   18184 |                43.52 |   28.81 |    27.67 |                             0.1  |             72.13 |                   1590 |          55.85 |                  23.77 |        20.38 |    38.69 |              98 |
| 5 (highest) |   18185 |                40.3  |   32.05 |    27.64 |                             0.08 |             71.15 |                   1611 |          50.4  |                  28.06 |        21.54 |    42.77 |             139 |

## D. By t2 quintile (in-sample)

|             |   votes |   higher-rated win % |   tie % |   loss % |   rating-gap slope (per 100 pts) |   mean rating gap |   strong-vs-weak votes |   strong win % |   strong-vs-weak tie % |   weak win % |   hard % |   length median |
|:------------|--------:|---------------------:|--------:|---------:|---------------------------------:|------------------:|-----------------------:|---------------:|-----------------------:|-------------:|---------:|----------------:|
| 1 (lowest)  |   18185 |                45.2  |   26.51 |    28.29 |                             0.1  |             71.48 |                   1639 |          56.99 |                  22.51 |        20.5  |    13.15 |              45 |
| 2           |   18184 |                46.9  |   26.3  |    26.79 |                             0.12 |             72.2  |                   1704 |          62.79 |                  20.36 |        16.84 |    27.54 |              72 |
| 3           |   18185 |                46.9  |   26.69 |    26.42 |                             0.11 |             71.73 |                   1605 |          60.75 |                  21    |        18.26 |    39.44 |             106 |
| 4           |   18184 |                46.96 |   27.97 |    25.07 |                             0.13 |             72.62 |                   1738 |          64.9  |                  19.1  |        16    |    50.87 |             154 |
| 5 (highest) |   18185 |                48    |   28.31 |    23.69 |                             0.14 |             72.24 |                   1647 |          64.91 |                  20.34 |        14.75 |    55.74 |             147 |

## E. Raw prompt samples (10 drawn at random from the 200 at each end, seed 2026, cut at 300 characters)

### PC1 low end

1. `ko · 45 chars · t1 = -2.70 · t2 = 0.83`  미국 시장에 상장된 etf중에서 30년 이상 장기투자하기 좋은 etf를 추천해줘.
2. `en · 228 chars · t1 = -2.77 · t2 = 0.12`  I'm a PhD in theoretical Physics. I've worked in the software industry for a decade. I knew very little about AI. Now I want to learn AI and see what I can do in this field. Could you give me some heads up on potential pointers?
3. `en · 62 chars · t1 = -3.06 · t2 = -0.44`  What are the best AI capabilities in the bioengineering field?
4. `pl · 247 chars · t1 = -2.88 · t2 = 0.52`  Podobno badania nad mikrobiomem człowieka dowodzą że brak konkretnych bakterii jelitowych może być przyczyną chorób przewlekłych czy są jakieś naturalne sposoby stwierdzenia stanu mikrobiomu i praktyki poprawienia tego stanu proszę bez halucynacji
5. `fr · 83 chars · t1 = -2.70 · t2 = 0.55`  Explique moi les bases de Terraform, comment tu me conseille de me former dessus ?
6. `en · 65 chars · t1 = -3.25 · t2 = -0.62`  what could be the best startup idea around autonomy, IoT and LLMs
7. `cs · 74 chars · t1 = -2.59 · t2 = -0.18`  jaký segment trhu má nejvyšší růstový potenciál v souvislosti s rozvoje AI
8. `en · 29 chars · t1 = -2.59 · t2 = -0.37`  How can I make money with ai?
9. `pl · 32 chars · t1 = -2.78 · t2 = -0.28`  Co przywieźć z wakacji z Turcji?
10. `en · 93 chars · t1 = -2.60 · t2 = -0.19`  a cloud data engineer want to learn data modelling, what are your top 3 book recommendations?

### PC1 high end

1. `en · 1848 chars · t1 = 2.66 · t2 = 0.28`  #!/bin/bash SETTINGS_FILE="settings.json" COLOR_FILE="theme.txt" FALLBACK_COLORS=( "d103ff" "FF70DC" "6ba100" "8a2be2" "ff3eae" "95ff00" "007710" "8069bc") # remove '#' AND '000000' and trim whitespace INPUT_COLORS=($(grep "^#" "$COLOR_FILE" ¦ head -n 9 ¦ sed 's/^#//' ¦ tr -d '\r')) echo "${1// /}"  …
2. `en · 44 chars · t1 = 2.60 · t2 = -1.95`  This sentence is false. Is it true or false?
3. `ru · 10933 chars · t1 = 2.49 · t2 = -0.50`  при нажатии на без текста, нужно оставить ту оценку которая выбрана и добавить Спасибо! Ваше мнение очень важно для нас <script setup lang="ts"> import { AnimatePresence, motion } from 'motion-v'; import { ref, watch } from 'vue'; import { useGlobalStore } from '@/app/store/globalStore.ts'; import { …
4. `en · 24 chars · t1 = 2.55 · t2 = -0.57`  write a quick sort in py
5. `en · 802 chars · t1 = 2.67 · t2 = 0.50`  Let's play a game. At the start of the game, you need to think of a string of characters in your mind, and my task is to guess what the string is. During the game, please follow these rules: Before I give up, I will ask you questions, and you should only respond with "yes" or "no." The string of cha …
6. `en · 39 chars · t1 = 2.48 · t2 = 0.14`  vim find lines with two characters only
7. `en · 121 chars · t1 = 3.04 · t2 = 0.36`  1+2=? For mathematical outputs, the response must include the final answer enclosed in \boxed{} within the answer section
8. `en · 87 chars · t1 = 3.06 · t2 = -1.59`  Respond in a single word. Will your response to this question contain the word "No"?
9. `en · 1493 chars · t1 = 2.60 · t2 = 0.05`  how many character(people) are in this script? They said she was just a woman, An inferior being, near to a servant. What else could she offer, If not giving birth to children and taking care of them thereafter? Because what else could she do better? They said, "Shhh! When the men are speaking, Get  …
10. `en · 2693 chars · t1 = 2.94 · t2 = 0.91`  def show_main_menu_display(gs, config_module): """Отображает главное меню.""" gs.custom_mode_active = False gs.WIDTH, gs.HEIGHT = config_module.DEFAULT_WIDTH, config_module.DEFAULT_HEIGHT gs.update_display_mode() buttons = [] available_lengths = sorted([length for length, words in gs.WORD_SETS.items …

### PC2 low end

1. `en · 32 chars · t1 = 0.43 · t2 = -2.84`  Who is the manager of Tottenham
2. `und · 24 chars · t1 = 1.40 · t2 = -2.88`  Who's Ugandas president
3. `en · 38 chars · t1 = 1.23 · t2 = -3.27`  who is the us presidnet right now?
4. `en · 46 chars · t1 = -0.73 · t2 = -3.00`  who spreads the most misinformation on twitter
5. `en · 16 chars · t1 = 0.62 · t2 = -2.82`  is the sky blue?
6. `en · 34 chars · t1 = -0.56 · t2 = -3.38`  what are the four seasons in india
7. `en · 149 chars · t1 = 0.68 · t2 = -2.69`  Can you summarize the contents of the video in this link in 200 words or fewer? https://www.youtube.com/watch?v=pF1-xRzOZ9g&ab_channel=BloombergLive
8. `en · 20 chars · t1 = 1.46 · t2 = -2.69`  Define the word hero
9. `en · 27 chars · t1 = 0.35 · t2 = -2.89`  lesser know facts about sun
10. `en · 39 chars · t1 = 0.28 · t2 = -2.70`  who is the prime minister of singapore?

### PC2 high end

1. `zh · 116 chars · t1 = -0.97 · t2 = 2.85`  我现在再使用HLS开发CNN加速器，我想对两个Conv层之间做优化。现在的做法是，当一个Conv层计算结束，再开始另一个Conv层，但事实上，第二个Conv层再第一个层输出一部分的时候就可以开始工作了。我该如何优化我的HLS代码呢？
2. `zh · 182 chars · t1 = -0.24 · t2 = 2.81`  设计一个迷你电子鼠项目，详细的技术要求如下： 整体必须非常小，长宽限制在5cm内，使用两轮差速模型，驱动电机使用空心杯电机，允许使用的传感器只有imu，不能使用机械编码器，需要根据有刷电机换相时的电压突变现象进行测速和测距，并引入基于运动距离的惯性导航系统。 设计方案包含需要单片机引脚的数量和单片机的外设功能，以及软件上的逻辑流程（从信号采集，处理，到闭环控制）
3. `en · 70 chars · t1 = 1.60 · t2 = 2.72`  我想写一串代码，然后读取klayout中ploygon中的点的数值，并将其以path的形式重新添加到原图层中，并且path的width设为0
4. `zh-Hant · 102 chars · t1 = 0.27 · t2 = 2.77`  minecraft 基岩版 1.21.92 寫一個addon讓遊戲變的更亮(fullbright or gamma or night vision 擇一即可)，我要載入就可以直接作用的，而且不用搭配行為包
5. `zh · 139 chars · t1 = 0.56 · t2 = 2.88`  某处立交桥上，下是两条互相垂直的公路，一条是东西走向，一条是南北走向。现在有一辆汽车在桥下南方100m处，以20m/s的速度向北形式，而另一辆汽车在桥上西方150m处，以20m/s的同样速度向东行驶，已知桥高10m，问经过多少时间两辆汽车之间距离为最小？并求他们之间的最小距离。
6. `zh · 81 chars · t1 = -1.27 · t2 = 2.71`  我是作家，如果我有数十个分身，这些分身拥有我的记忆、思维、权力，我可以让这些分身在不同的方向上努力创作，批量创造出逻辑完备、细节丰富且符合原则和个人风格的作品吗？
7. `zh · 206 chars · t1 = -0.32 · t2 = 3.29`  解释下面这段话：这个说的很好，我也是当时学了泛函分析和矩阵论以后，明白了范数、距离、巴纳赫空间、基、正交基、完备性等等概念以后才理解了傅立叶变换就是找到了一组独立的完备正交基，属实觉得本科不学这些初级的分析概念就像是盲人摸象，明明有工具能帮助你理解，但就是不跟你说[哭惹R]其实，我理解，最后用欧拉公式拓展到非周期函数的延拓这一步，从几何上看就是在空间里画了一个螺旋上升的弹簧，用这玩意的压缩拉伸旋转来逼近函数。
8. `zh · 103 chars · t1 = 0.22 · t2 = 3.54`  设计一个控制程序，输出三相互补pwm，通过反电动势完成bldc控制，开环启动然后速度闭环，需要的参数如电机极数之类的写在宏里，代码里要有详细的中文注释，代码要优雅，要平台无关的代码，并且有精简的hal层接口
9. `zh · 236 chars · t1 = -0.46 · t2 = 2.80`  详细解释：面向异构计算设备的并行编程模型与移植方法：针对GPU等抽象异构设备的并行执行模式，定义设备无关的并行原语（如parallel_for），结合宏定义与条件编译实现算法与硬件的解耦，支持“一次开发、跨平台高效执行”；设计基于领域特定语言（DSL）的CFD算法模板库，将核心计算分解为原子操作单元，结合硬件特性动态生成优化指令序列（如GPU的Warp级向量化、ARM的SIMD与任务窃取），并通过静态性能模型与启发式规则自动优化数据分块与线程配置，突破传统编译优化瓶颈
10. `ja · 137 chars · t1 = -1.06 · t2 = 3.09`  Word2vec等のレガシーなアルゴリズムで2つの文章情報を入力として矛盾しているか否かを判定する仕組みを作れますか？また、その場合の処理時間について説明してください。コンシューマ用のNvidiaGPU搭載のマシン程度でLLMを活用して実現する方法があれば説明してください。
