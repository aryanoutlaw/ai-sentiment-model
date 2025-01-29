import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from inference import process_local_video
import tempfile
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🎭 Multimodal Sentiment Analysis")
st.markdown("""
    This application analyzes videos to detect emotions and sentiments using multimodal deep learning.
    Upload a video to get started!
""")

def create_emotion_timeline(predictions):

    data = []
    emotions_set = set()
    
    for pred in predictions:
        timestamp = (pred['start_time'] + pred['end_time']) / 2

        for emotion in pred['emotions']:
            emotions_set.add(emotion['label'])
            data.append({
                'Time': timestamp,
                'Emotion': emotion['label'],
                'Confidence': emotion['confidence'],
                'Text': pred['text']
            })
    
    df = pd.DataFrame(data)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for emotion in emotions_set:
        emotion_data = df[df['Emotion'] == emotion]
        fig.add_trace(
            go.Scatter(
                x=emotion_data['Time'],
                y=emotion_data['Confidence'],
                name=emotion.title(),
                mode='lines+markers',
                hovertemplate="<b>%{y:.1%}</b> confidence<br>Text: %{customdata}<extra></extra>",
                customdata=emotion_data['Text'],
                line=dict(width=2),
                marker=dict(size=8)
            )
        )
    
    fig.update_layout(
        title='Emotion Timeline Analysis',
        xaxis_title='Video Timeline (seconds)',
        yaxis_title='Confidence Score',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        showlegend=True,
        height=500,
        margin=dict(r=150) 
    )
    
    return fig

def create_sentiment_distribution(predictions):

    sentiments_data = {
        'positive': {'count': 0, 'confidence_sum': 0},
        'negative': {'count': 0, 'confidence_sum': 0},
        'neutral': {'count': 0, 'confidence_sum': 0}
    }
    
    total_segments = len(predictions)
    
    for pred in predictions:
        sentiment = pred['sentiments'][0] 
        sentiments_data[sentiment['label']]['count'] += 1
        sentiments_data[sentiment['label']]['confidence_sum'] += sentiment['confidence']
    

    labels = []
    counts = []
    avg_confidences = []
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
    custom_colors = []
    
    for sentiment, data in sentiments_data.items():
        if data['count'] > 0:
            labels.append(sentiment.title())
            counts.append(data['count'])
            avg_conf = data['confidence_sum'] / data['count']
            avg_confidences.append(avg_conf)
            custom_colors.append(colors[sentiment])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=labels,
            y=counts,
            name="Segment Count",
            text=[f"{(c/total_segments):,.1%}" for c in counts],
            textposition='auto',
            marker_color=custom_colors
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=avg_confidences,
            name="Avg Confidence",
            mode='lines+markers+text',
            text=[f"{c:.1%}" for c in avg_confidences],
            textposition='top center',
            line=dict(color='rgba(0,0,0,0.5)', width=2),
            marker=dict(size=10)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Sentiment Distribution and Confidence',
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update yaxes
    fig.update_yaxes(title_text="Number of Segments", secondary_y=False)
    fig.update_yaxes(title_text="Average Confidence", tickformat='.0%',
                     range=[0, 1], secondary_y=True)
    
    return fig

def main():
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])
    
    if uploaded_file:
        st.video(uploaded_file)
        
        if st.button("Analyze Video"):
            # Creating a temporary file to save the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            try:
                with st.spinner('Analyzing video...'):

                    results = process_local_video(video_path)
                    
                    if results['utterances']:

                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Emotion timeline
                            st.plotly_chart(create_emotion_timeline(results['utterances']), use_container_width=True)
                        
                        with col2:
                            # Sentiment distribution
                            st.plotly_chart(create_sentiment_distribution(results['utterances']), use_container_width=True)
                        

                        st.subheader("📝 Detailed Analysis")
                        for utterance in results['utterances']:
                            st.markdown(f"**Text**: {utterance['text']}")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### 😊 Emotions")
                                for emotion in utterance['emotions']:
                                    st.progress(emotion['confidence'])
                                    st.write(f"{emotion['label'].title()}: {emotion['confidence']:.2%}")
                            
                            with col2:
                                st.markdown("### 💭 Sentiments")
                                for sentiment in utterance['sentiments']:
                                    st.progress(sentiment['confidence'])
                                    st.write(f"{sentiment['label'].title()}: {sentiment['confidence']:.2%}")
                            
                            st.markdown(f"**Timestamp**: {utterance['start_time']:.2f}s - {utterance['end_time']:.2f}s")
                            st.markdown("---")
                    else:
                        st.error("No predictions were generated for this video. Please try another video file.")
            
            except Exception as e:
                st.error(f"An error occurred while processing the video: {str(e)}")
            
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)

if __name__ == "__main__":
    main()