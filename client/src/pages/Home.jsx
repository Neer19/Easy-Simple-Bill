import React from 'react'
import styles from './Home.module.css'
import { useState } from 'react'
import axios from 'axios'
import { baseURL } from '../constants/baseURL'
import Loader from '../components/Loader'

// import Error from '../components/Error/Error'
const Home = () => {
    const [selectedFile, setSelectedFile] = useState();
	const [isFilePicked, setIsFilePicked] = useState(false);
  
  const [loading, setLoading] = useState(false)
    const changeHandler = (event) => {
        setSelectedFile(event.target.files[0]);
        console.log(event)
		setIsFilePicked(true);
        // console.log(selectedFile)
    }
    
    const handleSubmission = async (event) => {
        // console.log("Clicked")
        event.preventDefault()
        if(isFilePicked){
            const url = `${baseURL}/uploadFile`
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('fileName', selectedFile.name);
            const config = {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            };
            // console.log(formData)
            try {
              setLoading(true)
              
              const {data} = await axios.post(url, formData, config)
              setLoading(false)
              const url1 = window.URL.createObjectURL(new Blob([data]))
              //downloading csv data
              const a = document.createElement('a')
              a.href = url1
              a.download = 'data.csv'
              a.click()
              window.URL.revokeObjectURL(url1);
              
              
            } catch (error) {
              alert(error.message)
            }
        }
        else{
           alert("Please chose file")
           
        }
        
    }
  return (
    <div className={styles.app}>
        <div className={styles.child}>
          <div className={styles.file}>
          <input type="file" name="file" accept="image/png, image/gif, image/jpeg" onChange={changeHandler} />
			<div>
				<button className={styles.btn} onClick={handleSubmission}>Submit</button>
			</div>
           
          </div>
        </div>
        {loading && <h2 className={styles.heading}>Dowload will began shortly....</h2>}
        {loading && <Loader />}
      </div>
  )
}

export default Home