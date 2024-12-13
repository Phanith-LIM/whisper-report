import marimo

__generated_with = "0.10.0"
app = marimo.App(
    width="full",
    app_title="Whisper Evaluation Model",
    auto_download=["html", "ipynb"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Whisper Evaluation Model

        I trained this model using a dataset with a total duration of 25 hours. A portion of the data was sourced from the [PhanithLIM/aakanee-kh](https://huggingface.co/datasets/PhanithLIM/aakanee-kh) dataset, which contains 445 records and has a total duration of 1 hour, 7 minutes, and 27 seconds.

        Further details on the training process, data preprocessing, and evaluation metrics are included below to provide more context on the model's performance and capabilities.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd 
    from jiwer import cer, wer
    return cer, mo, pd, wer


@app.cell(hide_code=True)
def _(mo):
    model_size = mo.ui.dropdown(options=["tiny", "base", "small"], value="tiny", label="model size: ")
    model_size
    return (model_size,)


@app.cell(hide_code=True)
def _(model_size, pd):
    path = f'dataset/{model_size.value}-25h.csv'
    df = pd.read_csv(path, index_col=False)
    # mo.ui.table(df, show_column_summaries=False)
    return df, path


@app.cell(hide_code=True)
def _():
    def process_content(content):
        symbols_to_remove = [
            '(', ')', '[', ']', '{', '}', '<', '>', 
            '“', '”', '‘', '’', '«', '»', ',',
            '「', '」', '『', '』', '▁', '-',
            '៖', '។', '៛', '៕', '!', '​', '–', ' ', ''
        ]
        for symbol in symbols_to_remove:
            content = content.replace(symbol, '')
        return content
    return (process_content,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Character Error Rate""")
    return


@app.cell(hide_code=True)
def _(cer, df, mo, process_content):
    df['Actual'] = df['Actual'].apply(lambda x: process_content(x))
    df['Predict'] = df['Predict'].apply(lambda x: process_content(x))
    df['CER %'] = df.apply(lambda x: round(cer(x['Actual'], x['Predict']) * 100, 2), axis=1)
    # mo.ui.table(df, show_column_summaries=False)

    mean_cer = df['CER %'].mean()
    min_cer = df['CER %'].min()
    max_cer = df['CER %'].max()

    mo.vstack(
        [
            mo.md(f'Mean CER: {mean_cer:.2f}%'), 
            mo.md(f'Min CER: {min_cer}%'), 
            mo.md(f'Max CER: {max_cer}%')],
        align='start',
    )

    # print(f'Mean CER: {mean_cer:.2f}%')
    # print(f'Min CER: {min_cer}%')
    # print(f'Max CER: {max_cer}%')
    return max_cer, mean_cer, min_cer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Result""")
    return


@app.cell(hide_code=True)
def _(df, mo):
    slider = mo.ui.slider.from_series(df['CER %'], step=10, show_value=True)
    return (slider,)


@app.cell(hide_code=True)
def _(mo, slider):
    mo.hstack([slider, mo.md(f"Has value: {slider.value} %")], align='stretch', gap=50, widths=[5,2])
    return


@app.cell(hide_code=True)
def _(df, mo, slider):
    # filter_data = df[(df['CER %'] >= min) & (df['CER %'] <= max)]
    filter_data = df[df['CER %'] <= slider.value]
    filter_data = filter_data.sort_values('CER %')
    filter_data = filter_data.drop(columns=['Path', "Sampling Rate"])
    filter_data = filter_data.reset_index() 
    filter_data = filter_data.drop(columns=['index'])
    mo.ui.table(filter_data, show_column_summaries=False, pagination=True, label="Result of CER", wrapped_columns=['Actual', 'Predict'])
    return (filter_data,)


if __name__ == "__main__":
    app.run()
