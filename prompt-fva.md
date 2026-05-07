**FVA - Medium Article**
```markdown

Forecast Value Added
Better forecasting will allow your supply chain to face fewer shortages, more sales, less useless inventory, and streamlined operations. Ultimately, more profits and lower costs. In this article, I will show you how to improve your forecasting accuracy and reduce the workload of your teams by using the Forecast Value Added Framework. One stone, two birds.
Nicolas Vandeput
Nicolas Vandeput
·
Follow
11 min read
·
Dec 14, 2021
89

1



Credit
As I will show you in Conclusion and Next Steps at the end of the article, Forecast Value Added (FVA) doesn’t require massive investment. Its ROI will likely outshine any other improvement project. Forecast Value Added is simply waiting for a supply chain hero to lead its supply chain through this journey. Will this person be you?
Let’s imagine the following scenario: You are managing the demand forecasting process of your supply chain. It is a global supply chain selling products across multiple countries and regions. First, you use forecasting software to populate a baseline forecast. Then multiple teams provide inputs: First your demand planning team, then salespeople, and finally, there is a consensus meeting held where the final forecast number is negotiated agreed upon.

Figure 1 Your demand forecasting process (Source)
Your team achieves a noted forecast accuracy. However, you have no idea if you could do better, or if your forecasting software is doing a good job. You’re also worried that the consensus meeting might be more about politics and budget adherence than demand forecasting. Moreover, you suspect that your sales team is over forecasting on purpose to avoid shortages.
Is your forecasting model properly set up? Is the sales team creating bias? Does the consensus exercise increase or decrease the final accuracy?
Forecast Value Added
Demand forecasting is a resource-intensive process that raises two fundamental questions:
Does every stage in this process improve the overall accuracy?
Is the extra accuracy worth the burden?
As various actors input many numbers to come up with the final version, it is difficult to know who adds value and who does not. The ownership and accountability for the forecast are likely to get diluted during the process.
As the demand forecasting process owner, you need to ensure two things:
Efficacy. Each human edition of the forecast should make it more accurate, not less accurate. You want people to make the forecast more accurate.
Efficiency. You do not want your team to spend too much time working on the forecast. There is a point of diminishing returns: at some point, the time needed to improve your predictions further will not be worth the business value of the extra forecasting accuracy. Basically, there is no point in discussing changing a product’s forecast by 0.1% for two days.
To track efficacy (Are we improving the forecast?) and efficiency (Are we making good use of our time?), we will use the Forecast Value Added framework (FVA).
The idea of the Forecast Value Added framework (FVA)[1] is to track the accuracy of each step in a forecasting process (model, planners, sales team, consensus).[2] And give each team an FVA score based on the added accuracy they achieved compared to the previous step.
[1] Initially introduced in Gilliland, M. (2002). Is forecasting a waste of time? Supply Chain Management Review
[2] You should also track customer forecasts: they are too often trusted but do not always add any value. Identifying errors is a great opportunity for customer engagement and alignment.

Figure 2 Forecasting Value Added (Source: my demand forecasting training)
FVA is key to process excellence. Performing an FVA analysis of the whole forecasting process (demand planners, salespeople, senior management; you can even track the FVA of each individual separately) will ensure that every team will be the owner of their predictions and accountable for the achieved accuracy.
Ownership, accountability, and analytics are the perfect combination for a data-driven process aiming at excellence. That’s what FVA is about.
Forecast Value Added: Concept and Case Studies

Share

Watch on
In these examples, we are using MAE instead of MAPE. Tracking MAPE to measure forecast error is a bad practice. However, due to organizations’ tendency to stay with what has worked, it is more important to emphasize an adoption of FVA first rather than potentially alienate them by placing undue importance on the metric used.
Learn more about using MAE over MAPE here:
Forecast KPI: RMSE, MAE, MAPE & Bias
The article below is an extract from my book Data Science for Supply Chain Forecast, available here. You can find my…
towardsdatascience.com
Judgmental Biases
As you track the FVA of each step of the forecasting process, you will have the right analytics to reduce most judgmental bias (either intentional or unintentional).
As an example, let’s continue with the process outlined in the table above:
Demand planners are the first to update the forecast baseline. They managed to remove 2 points of MAE and improve the bias by another 2 points.
Usually, trained planners do a good job of improving a forecast baseline by gathering information from different sources (salespeople, marketing, customers) and forecasting product introduction (whereas models usually struggle with product life cycles).
The sales team comes into action next. Unfortunately, they worsen the forecast by increasing the mean absolute error by 2 points and the bias by 4.
Problem The sales team might be prone to intentional judgmental bias. In this example, they are inputting a high forecast to secure the inventory.
Solutions Sales need to work together with supply chain planners to find ways to secure inventory without biasing the forecast. This collaboration should be formalized within the S&OP cycle. Moreover, as salespeople couldn’t improve the forecast accuracy, they should spend less time updating the forecast and focus on the few products they are the surest about.
Fighting Intentional Biases. Individuals intentionally biasing their forecasts usually do this due to a misalignment of incentives. If you push your teams to deliver 100% service level, they will tend to over forecast demand. If you push your teams to overdeliver, they will tend to under forecast demand.
The last step of the process is senior management’s final validation.
Problem Senior management can also be biased to please the shareholders/board or to stick to the initial budget.
Solutions Senior management may be updating the forecast with what they hope to sell and not what they expect to sell. Therefore, they should note differences between
Forecast: how much we think the demand will be,
Plan: how much we should (or can) produce,
Budget: how much we agreed upon last year.
Like the sales team, senior managers’ modifications did not add accuracy. They should spend less time tweaking the forecast and not aiming for a specific demand target. Without a data-driven FVA analysis, it will be challenging to convince senior management to perform fewer forecast editions.

Contact us here
Process Efficiency
Forecast value add, is meant to ensure that each team in the process adds value, compared to the previous one (efficacy), and that they didn’t spend too much time doing it (efficiency). To measure efficiency, we will have to track roughly the time spent on each step in the forecasting process.

Figure 3 FVA with time spent
With the help of FVA, you will quickly realize that the marginal improvement of each new team working on the forecast is decreasing. It might be easy to improve the most significant shortcomings of a forecasting model (like product introductions). However, it is much more challenging to improve a forecast that has already been reviewed by professional teams relying on multiple sources of information.
Past a certain point, working more on the forecast will not be worth it. By tracking both the time spent and the added value, FVA will help you allocate just the right amount of resources to your process.

This article is inspired by the latest chapter of my book. You can read an 80 pages extract here.
Best Practices
Let’s review a few extra best practices when using the forecast value added framework.
FVA process analysis should be performed over multiple forecast cycles. Anyone can be (un)lucky from time to time, don’t rush to conclusions. The objective of FVA is to help management bring the best out of their teams. Don’t overreact to a few negative FVA rounds by ditching parts of the forecast process. The point is to find the root causes of the under-performance and fix the process (most likely by removing biases, promoting ownership, and aligning incentives). Removing steps of the process that should, in theory, add value is the last resort.
Do not hesitate to track and report FVA by product group, channel, or region. Especially if different sales channels (or business units) imply different information, teams, and buying behaviors.
If you want to push FVA further and focus on the most critical items, you should use it together with weighted KPIs.
More tips and best practices can be found in the conclusions drawn by Fildes and Goodwin (2007), who investigated the demand planning process of 4 British companies.
Do not spend time on minor adjustments. They are most likely within the error margin. They saw that planners were making numerous small adjustments to the forecasts, bringing nearly no added value and consuming time. (The need to act is a common cognitive bias).
Focus on larger adjustments. They are more likely to improve accuracy as senior management requires more explanations, and they carry higher (personal) risks if they are wrong.
Track the number of positive and negative adjustments. Planners tend to be overly optimistic (another usual cognitive bias), resulting in too many positive adjustments. Fildes and Goodwin noted that most positive adjustments decreased the accuracy, while most negative adjustments positively impacted accuracy. This can be explained easily: it takes more courage and data to bring bad news than good news. The authors even provocatively suggested banning positive adjustments altogether.
Should your Sales Team be Responsible for Forecasting?
Many supply chains and SME businesses rely on their sales team to generate sales forecasts. But is this a good…
nicolas-vandeput.medium.com
What Is a Good Forecast Error?
When teaching students or training professionals on forecasting KPIs, I like to repeat over and over the same question: “How do you know if a forecast is good enough?”
In the graph below, you can see the historical sales of a product and a forecast populated by a statistical model. This forecast achieved a historical forecast error of 36.1%.

Are these results above in Figure 4 good? Would you be satisfied if your forecasting tool showed you a forecast error of 36.1%?
The accuracy of a model depends on the demand’s inner complexity and random nature. Forecasting the number of smartphones sold in a country per month is much easier than predicting the sales of a specific phone model, at a particular store, during one specific day. It is exceedingly difficult to estimate a priori what is a reasonable forecast error for a particular dataset.
Demand Forecasting Best Practices
Demand Forecasting Best Practices: 9781633438095: Computer Science Books @ Amazon.com
www.amazon.com
Benchmarking
To know if a certain level of accuracy is good or bad, you must compare it against a benchmark (a simple model).
The idea is to compare your forecasting process against a benchmark to see by how much extra accuracy you can beat it. This comparison will tell you a lot about process efficiency: how much investment (in software and time) do you need to beat a simplistic method?

Figure 5 We will compare your forecasting process against a benchmark
Personally, I like to use moving averages (of 3 or 6 months) as a benchmark.
Many practitioners (and software vendors) use naïve forecasts rather than moving averages. Beating a naïve forecast is too easy: don’t get fooled by anyone proclaiming to beat such a benchmark.

Figure 6 As you can see, our model is beating the benchmark. That’s good news.
Seasonal Benchmarks. If you face seasonal demand, you can use a seasonal naïve forecast as a benchmark. Instead of using the previous period to forecast the next one, you have to use the same-period-last-year demand to forecast each new period. You can also use a seasonal moving average by averaging multiple previous seasonal cycles
Industry Benchmarks?
Some practitioners (and companies such as Gartner) advise you to compare yourself against other companies to know if you are doing a good job at forecasting. I do not support this approach: it would be comparing pears and apples. You do not know how your competitors are tracking accuracy, nor what forecasting granularity and horizon they use. Moreover, different businesses follow different strategies with varying product portfolio sizes, sales channels, and promotion strategies. These differences will result in differences in forecasting accuracy between companies within the same industry.
Assessing Products’ Forecastability: Forecasting Benchmarks vs. COV
In this article, I will present you with 3 ways to assess the forecastability of your products (or any time series)…
nicolas-vandeput.medium.com
Time for Action
By making every team the owner of their own forecasts, the Forecast Value Added framework will help you to ensure that each team makes meaningful changes. In addition, looking at the bias and accuracy of each team will allow you to spot bad practices and problems.
Moreover, as you combine FVA with weighted KPIs, you will be sure that your teams will focus on the most valuable products.
How to Get Started?
FVA is simple to understand and the most powerful tool to improve your demand forecasting process. But it requires a lot of data as you need to track multiple teams across multiple forecast lags. Moreover, you’ll need to invest in automation to run it smoothly: you can’t spend days creating a monthly dashboard.
I would then advise two ways to get started:
You can easily compare your consensus forecast against a benchmark: Run a moving average of your past sales and compare it with your forecast. You should beat it by 5 to 10%. If not, you have room for improvement!
To run FVA at scale, you’ll need software or to create an automated data pipeline (it is technically possible with Python — but not straightforward). Unfortunately, tools such as SAP APO and IBP (and most forecasting software, unfortunately) do not offer these possibilities. On the other hand, newer online platforms such as SKU Science offer free trials that will allow you to load your consensus forecast and compare its baseline forecast (Disclaimer, I am one of SKU Science’s co-founders).
Acknowledgment
Leo Ducrot, Matthew Aiken, Wes van Kemenade, Jeff Carruthers, John Harkins, Michael Gilliland, Jonathon Karelse, Farayi Kambarami

```


## Result

Summary — FVA addition

  New scoring functions in src/m5/scoring.py
  - fva_scores(inp, *, baseline, metric) — star mode: every model vs one baseline, returns DataFrame with fva_abs and
  fva_pct. Sorted "best value-add first".
  - fva_chain(inp, *, chain, metric) — chain mode: each step's improvement vs the previous step (Naive → Stats → LGBM →
  Ensemble). First row is baseline (FVA = 0).
  - fva_per_fold(inp, *, baseline, metric) — per-(model, cutoff) FVA so a single fold can't drive the conclusion
  (Vandeput's strongest practical recommendation).
  - Metric basis: mae (Vandeput's default), rmse, smape, wrmsse — all weighted by the M5 dollar-weights so FVA stays
  consistent with the rest of the leaderboard.

  Three new figures (13–15)
  - fig_fva_star — horizontal bars, green = adds value, red = destroys, with +0.xxxx (+x.x%) annotations.
  - fig_fva_waterfall — chain waterfall: each bar is the incremental delta vs prior step. Green/red coloring for value
  added/destroyed.
  - fig_fva_per_fold — strip plot with zero line; dots = per-fold FVA per model. Models whose dots straddle zero are
  "lucky-or-not" rather than systematically value-adding.

  CLI
  m5 score --model stats --model lgbm \
           --fva-baseline SeasonalNaive \
           --fva-metric mae \
           --fva-chain SeasonalNaive,Theta,LGBM
  - --fva-baseline defaults to SeasonalNaive (canonical M5 baseline; auto-skips if the column isn't in the merged CV)
  - --fva-metric mae (Vandeput's preference) | rmse | smape | wrmsse
  - --fva-chain is optional; provide a comma-separated ordered list to render the waterfall

  Outputs land in reports/
  - metrics/fva_star.{parquet,csv}, metrics/fva_per_fold.parquet, metrics/fva_chain.parquet
  - figures/13_fva_star.{png,svg,pdf}, 14_fva_waterfall.*, 15_fva_per_fold.*
  - Report (md + html) gets new sections, an appendix table, and auto-generated insight callouts (e.g., "Most
  consistently value-adding across folds: LGBM (100% of folds positive)").

  Tests — 6 new FVA-specific unit tests + the smoke test now requires all three FVA figures to render. 103 total pass;
  ruff + mypy clean.
