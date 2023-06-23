import cld3
import pandas as pd

COLUMNS = ['tweet', 'likes', 'retweet_count', 'user_screen_name', 'user_description', 'user_followers_count']
LANG = 'en'
TW_USERNAME_REGEX = "@[a-zA-Z0-9_]{0,15}"
URL_REGEX = "\b(?:https?:\/\/|www\.)\S+\b"
SPACES_REGEX = "\s+"


def is_lang(row, lang='en'):
    prediction = cld3.get_language(row['tweet'])
    if prediction.language == lang and prediction.is_reliable:
        return True
    else:
        return False


def get_selected_columns(df, columns):
    return df[columns]


def delete_hashtag_symbol(df):
    df['tweet'] = df['tweet'].replace('#', '', regex=True)
    return df


def delete_twitter_username(df):
    df['tweet'] = df['tweet'].replace(TW_USERNAME_REGEX, '', regex=True)
    return df


def delete_urls(df):
    df['tweet'] = df['tweet'].replace(URL_REGEX, '', regex=True)
    return df


def delete_multiple_spaces(df):
    df['tweet'] = df['tweet'].replace(SPACES_REGEX, '', regex=True)


def filter_by_language(df, lang='en'):
    mask = df.apply(is_lang, axis=1)
    return df.loc[mask, :]


if __name__ == '__main__':
    trump_df = pd.read_csv('dataset/hashtag_donaldtrump_short.csv', sep=',')
    biden_df = pd.read_csv('dataset/hashtag_joebiden_short.csv', sep=',')

    trump_df = get_selected_columns(trump_df, COLUMNS)
    biden_df = get_selected_columns(biden_df, COLUMNS)

    # Para filtrar por len
    trump_df.loc[trump_df['tweet'].str.len() < 50]['tweet']

    # Deleting the 'Hashtag' symbol but keeping the content
    trump_df['tweet'] = trump_df['tweet'].replace('#', '', regex=True)
    biden_df['tweet'] = biden_df['tweet'].replace('#', '', regex=True)

    # Deleting twitter usernames when replying or mentioning someone
    trump_df['tweet'] = trump_df['tweet'].replace(TW_USERNAME_REGEX, '', regex=True)
    biden_df['tweet'] = biden_df['tweet'].replace(TW_USERNAME_REGEX, '', regex=True)

    # Deleting URLs
    trump_df['tweet'] = trump_df['tweet'].replace(URL_REGEX, '', regex=True)
    biden_df['tweet'] = biden_df['tweet'].replace(URL_REGEX, '', regex=True)

    # Deleting spaces
    trump_df['tweet'] = trump_df['tweet'].replace(SPACES_REGEX, ' ', regex=True)
    biden_df['tweet'] = biden_df['tweet'].replace(SPACES_REGEX, ' ', regex=True)

    trump_df = filter_by_language(trump_df, LANG)
    biden_df = filter_by_language(biden_df, LANG)

    # Create the pipeline
    # filter_by_lang_partial = partial(filter_by_language, lang=LANG)
    # pipeline = [delete_hashtag_symbol, delete_twitter_username, delete_urls, delete_leading_and_trailing_spaces, filter_by_lang_partial]

    # Apply the pipeline to the DataFrame
    # trump_df = trump_df.pipe(lambda x: x.pipe(*pipeline))
    # biden_df = biden_df.pipe(lambda x: x.pipe(*pipeline))
