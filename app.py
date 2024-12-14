import marimo

__generated_with = "0.10.2"
app = marimo.App(width="full", app_title="Whisper Evaluation Model")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Whisper Evaluation Model

        I used the [PhanithLIM/aakanee-kh](https://huggingface.co/datasets/PhanithLIM/aakanee-kh) dataset for evaluation, which contains 445 records and has a total duration of 1 hour, 7 minutes, and 27 seconds. The evaluation was done using Character Error Rate (CER).
        Further details on the training process, data preprocessing, and evaluation metrics are included below to provide more context on the model's performance and capabilities.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    from jiwer import cer, wer
    return cer, mo, os, pd, plt, wer


@app.cell(hide_code=True)
def _(mo, os):
    files = os.listdir('dataset')
    models = [file.split('-')[0] for file in files]
    versions = [file.split('-')[1] for file in files]
    durations = [file.split('-')[2].split('.')[0] for file in files]
    model_size = mo.ui.dropdown(options=models, value=models[0], label="model size: ")
    version = mo.ui.dropdown(options=versions, value=versions[0], label="version: ")
    duration =  mo.ui.dropdown(options=durations, value=durations[0], label="duration: ")
    return duration, durations, files, model_size, models, version, versions


@app.cell(hide_code=True)
def _(duration, mo, model_size, version):
    columns = mo.vstack(
        [
            model_size, 
            version,
            duration,
        ],
    )
    columns
    return (columns,)


@app.cell(hide_code=True)
def _(duration, model_size, pd, version):
    path = f'dataset/{model_size.value}-{version.value}-{duration.value}.csv'
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
            mo.md(f'Max CER: {max_cer}%'),
            mo.md(f'Records: {len(df.index)}'),
        ],
        align='start',
    )

    # print(f'Mean CER: {mean_cer:.2f}%')
    # print(f'Min CER: {min_cer}%')
    # print(f'Max CER: {max_cer}%')
    return max_cer, mean_cer, min_cer


@app.cell
def _(mo):
    mo.md(r"""## Visualizing Models""")
    return


@app.cell(hide_code=True)
def _(cer, pd, process_content):
    def matrix_model(data: pd.DataFrame):  # Ensure the argument is a DataFrame
        data['Actual'] = data['Actual'].apply(lambda x: process_content(x))
        data['Predict'] = data['Predict'].apply(lambda x: process_content(x))
        data['CER %'] = data.apply(lambda x: round(cer(x['Actual'], x['Predict']) * 100, 2), axis=1)

        # Calculate statistics for CER
        mean_cer = data['CER %'].mean()
        min_cer = data['CER %'].min()
        max_cer = data['CER %'].max()

        return mean_cer, min_cer, max_cer
    return (matrix_model,)


@app.cell(hide_code=True)
def _(files, matrix_model, pd, version):
    version_files = [file for file in files if version.value in file]
    body = []
    for v in version_files:
        df_model = pd.read_csv(f'dataset/{v}')
        mean, min_val, max_val = matrix_model(df_model) 
        body.append(
            {
                "model": v.split('-')[0],
                "mean": mean,
                "min": min_val,  # Correctly assign the minimum value
                "max": max_val   # Correctly assign the maximum value
            }
        )
    return body, df_model, max_val, mean, min_val, v, version_files


@app.cell(hide_code=True)
def _(body, mo, plt):
    models_data = [entry['model'] for entry in body]
    mean_values = [entry['mean'] for entry in body]
    min_values = [entry['min'] for entry in body]
    max_values = [entry['max'] for entry in body]

    # Bar colors for each category
    bar_colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Create the figure and axes for 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Mean values
    axes[0].bar(models_data, mean_values, color='tab:blue', alpha=0.7)
    axes[0].set_title('Mean Values')
    axes[0].set_ylabel('Mean')

    # Plot Min values
    axes[1].bar(models_data, min_values, color='tab:orange', alpha=0.7)
    axes[1].set_title('Min Values')
    axes[1].set_ylabel('Min')

    # Plot Max values
    axes[2].bar(models_data, max_values, color='tab:green', alpha=0.7)
    axes[2].set_title('Max Values')
    axes[2].set_ylabel('Max')

    # Add labels to the x-axis
    for ax in axes:
        ax.set_xlabel('Model')

    # Adjust layout to prevent overlap
    mo.mpl.interactive(plt.gcf())
    return (
        ax,
        axes,
        bar_colors,
        fig,
        max_values,
        mean_values,
        min_values,
        models_data,
    )


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
    filter_data = filter_data.sort_values('CER %', ascending=True)
    filter_data = filter_data.drop(columns=['Path', "Sampling Rate"])
    filter_data = filter_data.reset_index() 
    filter_data = filter_data.drop(columns=['index'])
    mo.ui.table(filter_data, show_column_summaries=False, pagination=True, label="Result of CER", wrapped_columns=['Actual', 'Predict'])
    return (filter_data,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
