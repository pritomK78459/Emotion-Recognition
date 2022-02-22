import React, { Fragment, useState } from "react";

function Upload() {


    const [selectedFile, setSelectedFile] = useState();     // File for sending to the api endpoint
    const [previewFile, setPreviewFile] = useState();       // File for preview
    const [isFilePicked, setIsFilePicked] = useState(false); 
    const [resultEmotion, setResultEmotion] = useState();   // Predicted emotion returned from the api
    const [emotionConfidence, setEmotionConfidence] = useState();   // Confidence score of the predicted emotion from the api
    const [isPredicted, setIsPredicted] = useState(false);   

    const changeHandler = (event) => {

        // store the image uploaded

        setSelectedFile(event.target.files[0]);
        setPreviewFile(URL.createObjectURL(event.target.files[0]));
        setIsFilePicked(true);
    }

    const handleSubmission = () => {

        // submit the form with body as file with content as image

        const formData = new FormData();

        formData.append('file', selectedFile);
        console.log(formData)

        fetch(
            'http://localhost:8000/predict',    // API endpoint
            {
                method: 'POST',
                body: formData,
            }
        ).then((response) => response.json()).then((result) => {
            console.log('Success: ', result);
            setResultEmotion(result.class);
            setEmotionConfidence(result.confidence);
            setIsPredicted(true);
           
        })
    }

    return (
        <Fragment>
            <div className="py-10 bg-gray-200 px-2">
                <div className="max-w-full mx-auto rounded-lg overflow-hidden md:max-w-lg">
                    <div className="md:flex">
                        <div className="w-full p-3">
                            <div className="relative border-dotted h-48 rounded-lg border-dashed border-2 border-blue-700 bg-gray-100 flex justify-center items-center">
                                <div className="absolute">
                                    <div className="flex flex-col items-center"> <i className="fa fa-folder-open fa-4x text-blue-700"></i> <span className="block text-gray-400 font-normal">Attach you files here</span> </div>
                                </div> <input type="file" className="h-full w-full opacity-0" name="" onChange={changeHandler} />
                            </div>
                            <div>
                                <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" onClick={handleSubmission}> Upload </button>
                            </div>

                            <div>
                                {
                                    isPredicted ? (
                                        <div className="max-w-sm rounded overflow-hidden shadow-lg">
                                            <img className="w-full" src={previewFile} alt="result"/>
                                            <div className="px-6 py-4">
                                                <div className="font-bold text-xl mb-2">Emotion: {resultEmotion}</div>
                                                <p className="text-gray-700 text-base">
                                                Confidence score: {emotionConfidence} 
                                                </p>
                                            </div>
                                            </div>
                                        
                                    ): (
                                        <p>No File Selected</p>
                                    )
                                }
                                
                            </div>
                            
                        </div>
                    </div>
                </div>
            </div>
        </Fragment>
    )
}

export default Upload;