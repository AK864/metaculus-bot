import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)


class TemplateForecaster(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q2.
    Main template bot changes since Q1
    - Support for new units parameter was added
    - You now set your llms when you initialize the bot (making it easier to switch between and benchmark different models)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = 2  # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif os.getenv("EXA_API_KEY"):
                research = await self._call_exa_smart_searcher(
                    question.question_text
                )
            elif os.getenv("PERPLEXITY_API_KEY"):
                research = await self._call_perplexity(question.question_text)
            elif os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
            else:
                logger.warning(
                    f"No research provider found when processing question URL {question.page_url}. Will pass back empty string."
                )
                research = ""
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )
            return research

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around an search on Exa.ai
        """
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        response = await searcher.invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            You will be presented with an interview question, and you must make a prediction on the outcome of the question in order to earn points.
            There are other professional forecasters participating, and they are also earning points.
            Your goal is to maximize the likelihood of you earning more points than the other professional forecasters after forecasting about 1000 questions like this.
            Your score is based on a logarithmic scale, so going from 99% to 99.9% only gives you a tiny advantage if you are correct (+0.009), but a huge penalty if you are wrong (-2.3).
            Therefore, it may make sense to stay away from predicting extremely small percentages (<1%) and extremely large percentages (>99%) as a hedge to ensure that you can never lose an extremely large number of points when you make an error.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied.
            When there is confusion as to how a scenario will resolve, interpret the resolution criteria as literally as possible, giving the resolution criteria priority even over the question statement:

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before predicting, you must write the following:
            (1a) The time left until the outcome to the question is known.
            (1b) The status quo outcome if nothing changed.
            (1c) The expectations of experts and markets.
            (1d) A brief description of all possible scenarios that would result in a No outcome and all possible scenarios that would result in a Yes outcome.

            Next, you must attempt to make a prediction just from data analysis. To do this, take the following steps as applicable and write your findings:
            (2a) Search the internet and news for past data points where a scenario, similar to the one asked about in the question, played out.
                 Try to find as many of these as possible, while ensuring that the scenarios are similar to the one in the question. Look back as many days, months, and years, as necessary, so long as the data remains relevant.
                 Keep in mind that data from 2020-2022 is often erratic due to the COVID pandemic, and therefore outliers in that time range can be disregarded if you deem it appropriate to do so.
            (2b) Compute a base rate for the question, if possible. If the question is asking if something will occur, figure out how many times something similar has occurred in the past, and over what time interval.
                 This should imply a base rate (occurrences/length of time). Only do this if you have found enough data to compute a meaningful base rate.
            (2c) If you have found numerical historical data, determine if there are any statistically significant patterns in it.
                 The data may be trending linearly or exponentially, or there may be cyclical repetition in the data on certain days of the week or seasons in the year.
            (2d) If the question asks if a certain situation will resolve before a certain date, estimate the chance the situation will resolve within a smaller time frame, like a day,
                 and then calculate a probability estimate for the whole question from that probability.
                 For example, if the probability the question will resolve YES in a single day is p (as a decimal), then the probability it will resolve YES over the course of a question lasting 100 days is 1-(1-p)^100.
                 You will need to adjust this formula depending on the shortened time interval you have selected, and the length of the question until resolution.
                 If you think the probability of a YES resolution is different on different days, factor this into your prediction and don’t blindly use the above formula.
                 If you follow this method, ensure you are reasoning mathematically and multiplying probabilities together.

            If you deem the data analysis to be irrelevant to your prediction, you are allowed to disregard it, or simply use it as a low-weight prior to your prediction.

            Now, you must come up with a better prediction. To do this, for each scenario you listed in (1c), give a likelihood estimate, in percent, of that scenario occurring.
            If applicable, use the results from your data analysis, prioritizing the numbers you computed from larger samples of data.
            Make sure you construct scenarios that are mutually exclusive so that the sum of the likelihoods is equal to 1. Do not force yourself to predict percentages that are a multiple of 5 or 10. 

            Then, figure out the total probability of the question resolving YES.
            Make sure you write your rationale for your entire prediction.
            
            Then, randomly choose either the word “high” or “low.” Only follow the below instructions depending on the word you selected:

            If you chose the word “high”:
            I am telling you that you overlooked something and the prediction you told me is too high.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.
            Write out your rationale and make a new prediction.
            Then, I am telling you that you overlooked something again and the new prediction you told me is now too low.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.

            If you chose the word “low”:
            I am telling you that you overlooked something and the prediction you told me is too low.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.
            Write out your rationale and make a new prediction.
            Then, I am telling you that you overlooked something again and the new prediction you told me is now too high.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.

            Make one final adjustment to your prediction.
            This adjustment should account for the fact that you are fallible and may have made some incorrect assumptions, or your research or knowledge about the question was insufficient.
            This adjustment should also account for the possibility of edge cases and black swan events that have never occurred historically.
            This is your final prediction.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            You will be presented with an interview question, and you must make a prediction on the outcome of the question in order to earn points.
            There are other professional forecasters participating, and they are also earning points.
            Your goal is to maximize the likelihood of you earning more points than the other professional forecasters after forecasting about 1000 questions like this.
            Your score is based on a logarithmic scale, so going from 99% to 99.9% only gives you a tiny advantage if you are correct (+0.009), but a huge penalty if you are wrong (-2.3).
            Therefore, it may make sense to stay away from predicting extremely small percentages (<1%) and extremely large percentages (>99%) as a hedge to ensure that you can never lose an extremely large number of points when you make an error.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied.
            When there is confusion as to how a scenario will resolve, interpret the resolution criteria as literally as possible, giving the resolution criteria priority even over the question statement:

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before predicting, you must write the following:
            (1a) The time left until the outcome to the question is known.
            (1b) The status quo outcome if nothing changed.
            (1c) The expectations of experts and markets.
            (1d) A brief description of all possible scenarios that would result in each of the options to this question.

            Next, you must attempt to make a prediction just from data analysis. To do this, take the following steps as applicable and write your findings:
            (2a) Search the internet and news for past data points where a scenario, similar to the one asked about in the question, played out.
                 Try to find as many of these as possible, while ensuring that the scenarios are similar to the one in the question. Look back as many days, months, and years, as necessary, so long as the data remains relevant.
                 Keep in mind that data from 2020-2022 is often erratic due to the COVID pandemic, and therefore outliers in that time range can be disregarded if you deem it appropriate to do so.
            (2b) Compute a base rate for the question, if possible. If the question is asking if something will occur, figure out how many times something similar has occurred in the past, and over what time interval.
                 This should imply a base rate (occurrences/length of time). Only do this if you have found enough data to compute a meaningful base rate.
            (2c) If you have found numerical historical data, determine if there are any statistically significant patterns in it.
                 The data may be trending linearly or exponentially, or there may be cyclical repetition in the data on certain days of the week or seasons in the year.
            (2d) If the question asks if a certain situation will occur before a certain date, estimate the chance the situation will resolve within a smaller time frame, like a day,
                 and then calculate a probability estimate for the whole question from that probability.
                 For example, if the probability something will occur in a single day is p (as a decimal), then the probability it will occur over the course of a question lasting 100 days is 1-(1-p)^100.
                 You will need to adjust this formula depending on the shortened time interval you have selected, and the length of the question until resolution.
                 If you think the probability of a question outcome occurring is different on different days, factor this into your prediction and don’t blindly use the above formula.
                 If you follow this method, ensure you are reasoning mathematically and multiplying probabilities together.

            If you deem the data analysis to be irrelevant to your prediction, you are allowed to disregard it, or simply use it as a low-weight prior to your prediction.

            Now, you must come up with a better prediction. To do this, for each scenario you listed in (1c), give a likelihood estimate, in percent, of that scenario occurring.
            If applicable, use the results from your data analysis, prioritizing the numbers you computed from larger samples of data.
            Make sure you construct scenarios that are mutually exclusive so that the sum of the likelihoods is equal to 1. Do not force yourself to predict percentages that are a multiple of 5 or 10. 

            Then, figure out the total probability of the question resolving as each of the different question options.
            Make sure you write your rationale for your entire prediction.
            
            Then, randomly choose either the word “high” or “low.” Only follow the below instructions depending on the word you selected:

            If you chose the word “high”:
            I am telling you that you overlooked something and the prediction you told me is too confident and concentrated.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.
            Write out your rationale and make a new prediction.
            Then, I am telling you that you overlooked something again and the new prediction you told me is now too underconfident and spread out.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.

            If you chose the word “low”:
            I am telling you that you overlooked something and the prediction you told me is too underconfident and spread out.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.
            Write out your rationale and make a new prediction.
            Then, I am telling you that you overlooked something again and the new prediction you told me is now too confident and concentrated.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.

            Make one final adjustment to your prediction.
            This adjustment should account for the fact that you are fallible and may have made some incorrect assumptions, or your research or knowledge about the question was insufficient.
            This adjustment should also account for the possibility of edge cases and black swan events that have never occurred historically.
            This is your final prediction.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            You will be presented with an interview question, and you must make a prediction on the outcome of the question in order to earn points.
            Your prediction will be a cumulative probability distribution, so you must provide values for the 2nd, 5th, 10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th, 95th, and 98th percentiles.
            There are other professional forecasters participating, and they are also earning points.
            Your goal is to maximize the likelihood of you earning more points than the other professional forecasters after forecasting about 1000 questions like this.
            Your score is based on a logarithmic scale, so going from 99% to 99.9% only gives you a tiny advantage if you are correct (+0.009), but a huge penalty if you are wrong (-2.3).
            Therefore, it may make sense to stay away from predicting extremely small percentages (<1%) and extremely large percentages (>99%) as a hedge to ensure that you can never lose an extremely large number of points when you make an error.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied.
            When there is confusion as to how a scenario will resolve, interpret the resolution criteria as literally as possible, giving the resolution criteria priority even over the question statement:

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before predicting, you must write the following:
            (1a) The time left until the outcome to the question is known.
            (1b) The status quo outcome if nothing changed.
            (1c) The expectations of experts and markets.
            (1d) A brief description of all possible scenarios that would result in a No outcome and all possible scenarios that would result in a Yes outcome.

            Next, you must attempt to make a prediction just from data analysis. To do this, take the following steps as applicable and write your findings:
            (2a) Search the internet and news for past data points where a scenario, similar to the one asked about in the question, played out.
                 Try to find as many of these as possible, while ensuring that the scenarios are similar to the one in the question. Look back as many days, months, and years, as necessary, so long as the data remains relevant.
                 Keep in mind that data from 2020-2022 is often erratic due to the COVID pandemic, and therefore outliers in that time range can be disregarded if you deem it appropriate to do so.
            (2b) Compute a base rate for the question, if possible. If the question is asking if something will occur, figure out how many times something similar has occurred in the past, and over what time interval.
                 This should imply a base rate (occurrences/length of time). Only do this if you have found enough data to compute a meaningful base rate.
            (2c) If you have found numerical historical data, determine if there are any statistically significant patterns in it.
                 The data may be trending linearly or exponentially, or there may be cyclical repetition in the data on certain days of the week or seasons in the year.
            (2d) If the question asks if a certain situation will resolve before a certain date, estimate the chance the situation will resolve within a smaller time frame, like a day,
                 and then calculate a probability estimate for the whole question from that probability.
                 For example, if the probability the question will resolve YES in a single day is p (as a decimal), then the probability it will resolve YES over the course of a question lasting 100 days is 1-(1-p)^100.
                 You will need to adjust this formula depending on the shortened time interval you have selected, and the length of the question until resolution.
                 If you think the probability of a YES resolution is different on different days, factor this into your prediction and don’t blindly use the above formula.
                 If you follow this method, ensure you are reasoning mathematically and multiplying probabilities together.

            If you deem the data analysis to be irrelevant to your prediction, you are allowed to disregard it, or simply use it as a low-weight prior to your prediction.

            Now, you must come up with a better prediction. To do this, for each scenario you listed in (1c), give a likelihood estimate, in percent, of that scenario occurring.
            If applicable, use the results from your data analysis, prioritizing the numbers you computed from larger samples of data.
            Make sure you construct scenarios that are mutually exclusive so that the sum of the likelihoods is equal to 1. Do not force yourself to predict percentages that are a multiple of 5 or 10. 

            Then, figure out the total probability of the question resolving YES.
            Make sure you write your rationale for your entire prediction.
            
            Then, randomly choose either the word “high” or “low.” Only follow the below instructions depending on the word you selected:

            If you chose the word “high”:
            I am telling you that you overlooked something and most of the percentiles in the prediction you told me are too high.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.
            Write out your rationale and make a new prediction.
            Then, I am telling you that you overlooked something again and most of the percentiles in the prediction you told me are now too low.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.

            If you chose the word “low”:
            I am telling you that you overlooked something and most of the percentiles in the prediction you told me are too low.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.
            Write out your rationale and make a new prediction.
            Then, I am telling you that you overlooked something again and most of the percentiles in the prediction you told me are now too high.
            Figure out why, keeping in mind all of the research and data analysis you have already done, and adjust your prediction if you think it is necessary.

            Make one final adjustment to your prediction.
            This adjustment should account for the fact that you are fallible and may have made some incorrect assumptions, or your research or knowledge about the question was insufficient.
            This adjustment should also account for the possibility of edge cases and black swan events that have never occurred historically.
            This is your final prediction.

            The last thing you write is your final answer as:
            "
            Percentile 2: XX
            Percentile 5: XX
            Percentile 10: XX
            Percentile 20: XX
            Percentile 30: XX
            Percentile 40: XX
            Percentile 50: XX
            Percentile 60: XX
            Percentile 70: XX
            Percentile 80: XX
            Percentile 90: XX
            Percentile 95: XX
            Percentile 98: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="metaculus/anthropic/claude-3-7-sonnet-latest",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4.1",
        },
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
