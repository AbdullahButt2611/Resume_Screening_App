import streamlit as st
import pickle
import re
import nltk

nltk.download("punkt")
nltk.download('stopwords')

# Loading Models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))


def cleanResume(txt):
    cleanTxt = re.sub('http\S+\s', ' ', txt)
    cleanTxt = re.sub('RT|cc', ' ', cleanTxt)      # Reomving RT and CC
    cleanTxt = re.sub('#\S+\s', ' ', cleanTxt)
    cleanTxt = re.sub('@\S+', '  ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanTxt)    # Removing Special Characters
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)         # Removing Sequences
    return cleanTxt



# Web Application Code
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        clean_resume = cleanResume(resume_text)
        clean_resume = tfidfd.transform([clean_resume])
        prediction_id = clf.predict(clean_resume)[0]
        
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)



# Python Main function
if __name__=='__main__':
    main()