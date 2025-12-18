import streamlit as st
import requests
import pandas as pd
from io import StringIO

API_URL = "http://localhost:8000"  

st.set_page_config(
    page_title="Анализ текстов",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Анализ тем в текстах")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Настройки")
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            st.success("✅ API доступно")
        else:
            st.error("❌ API недоступно")
    except:
        st.warning("⚠️ Не могу подключиться к API")
    
    st.markdown("---")
    model_type = st.selectbox(
        "Выберите модель:",
        ["lda", "nmf", "bertopic"],
        index=0
    )
    
    show_words = st.checkbox("Показать топ-слова", value=True)

# Вкладки
tab1, tab2 = st.tabs(["📝 Один текст", "📁 Много текстов"])

with tab1:
    st.header("Анализ одного текста")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text = st.text_area(
            "Введите текст для анализа:",
            height=150,
            placeholder="Введите текст здесь..."
        )
    
    
    if st.button("🔍 Анализировать", type="primary"):
        if text and len(text) > 10:
            with st.spinner("Анализируем..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={
                            "text": text,
                            "model_type": model_type,
                            "return_probabilities": True,
                            "return_top_words": show_words
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        topic = result["main_topic"]
                        
                        # Показываем результат
                        st.success("✅ Анализ завершен!")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Тема ID", topic["topic_id"])
                            st.metric("Название", topic["topic_name"])
                        
                        with col_b:
                            if topic.get("probability"):
                                st.metric(
                                    "Вероятность",
                                    f"{topic['probability']:.1%}"
                                )
                        
                        # Топ-слова
                        if show_words and topic.get("top_words"):
                            st.subheader("🏷️ Топ-слова темы:")
                            words_html = " ".join(
                                [f"<span style='background-color:#e6f3ff; padding:5px 10px; margin:3px; border-radius:5px; display:inline-block;'>{word}</span>" 
                                 for word in topic["top_words"]]
                            )
                            st.markdown(words_html, unsafe_allow_html=True)
                        
                        # Исходный текст
                        with st.expander("📄 Исходный текст"):
                            st.text(text[:500] + ("..." if len(text) > 500 else ""))
                    
                    else:
                        st.error(f"Ошибка API: {response.text}")
                        
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
        else:
            st.warning("Введите текст (минимум 10 символов)")

with tab2:
    st.header("Анализ многих текстов")
    
    upload_option = st.radio(
        "Способ загрузки:",
        ["Загрузить файл", "Ввести вручную"]
    )
    
    texts = []
    
    if upload_option == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Загрузите файл (TXT или CSV)",
            type=["txt", "csv"]
        )
        
        if uploaded_file:
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if line.strip()]
            else:  # CSV
                df = pd.read_csv(uploaded_file)
                # Ищем колонку с текстом
                text_cols = [col for col in df.columns if 'text' in col.lower() or 'message' in col.lower()]
                if text_cols:
                    texts = df[text_cols[0]].dropna().astype(str).tolist()
                else:
                    st.warning("Не найдена колонка с текстом в CSV")
    
    else:  # Ввод вручную
        manual_text = st.text_area(
            "Введите тексты (каждый с новой строки):",
            height=200,
            placeholder="Текст 1\nТекст 2\nТекст 3"
        )
        if manual_text:
            texts = [line.strip() for line in manual_text.split('\n') if line.strip()]
    
    if texts:
        st.info(f"📊 Найдено текстов: {len(texts)}")
        
        if st.button("📈 Анализировать все", type="primary"):
            # Ограничиваем количество
            if len(texts) > 100:
                st.warning(f"Будет проанализировано только первые 100 из {len(texts)} текстов")
                texts = texts[:100]
            
            results = []
            progress_bar = st.progress(0)
            
            for i, text in enumerate(texts):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={
                            "text": text,
                            "model_type": model_type,
                            "return_probabilities": False,
                            "return_top_words": False
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            "Текст": text[:100] + ("..." if len(text) > 100 else ""),
                            "Тема ID": result["main_topic"]["topic_id"],
                            "Название темы": result["main_topic"]["topic_name"]
                        })
                
                except Exception as e:
                    results.append({
                        "Текст": text[:100] + "...",
                        "Тема ID": "Ошибка",
                        "Название темы": str(e)[:50]
                    })
                
                progress_bar.progress((i + 1) / len(texts))
            
            # Показываем результаты в таблице
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Статистика
                st.subheader("📊 Статистика")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Всего текстов", len(results))
                
                with col2:
                    unique_topics = df_results["Тема ID"].nunique()
                    st.metric("Уникальных тем", unique_topics)
                
                with col3:
                    if "Тема ID" in df_results.columns:
                        most_common = df_results["Тема ID"].mode()
                        if len(most_common) > 0:
                            st.metric("Самая частая тема", most_common[0])
                
                # Экспорт
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать результаты (CSV)",
                    data=csv,
                    file_name=f"results_{model_type}.csv",
                    mime="text/csv"
                )

# Информация в футере
st.markdown("---")
st.caption(f"Модель: {model_type.upper()} | API: {API_URL}")