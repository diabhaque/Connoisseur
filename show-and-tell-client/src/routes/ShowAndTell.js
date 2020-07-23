import React, { useContext, useState } from 'react'
import { Card, Upload, Icon, message, Input, Button, Spin } from 'antd'
import ImageContext from '../context/image.context'
import PhotoUpload from '../utils/PhotoUpload'
import 'antd/dist/antd.css';
import axios from 'axios'

const ShowAndTell = () => {
    const [image] = useContext(ImageContext)
    const [loading, setLoading] = useState(false)
    const [fetched, setFetched] = useState(undefined)

    const generateCaption=()=>{
        if(Object.keys(image).length>1){
            setLoading(true)
            setFetched(undefined)
            console.log('sending...')
            const data={
                image: image
            }
            axios.post("http://localhost:5000/predict", data,{
                headers: {  
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            }).then(res=>{
                console.log()
                setLoading(false)
                setFetched(res.data.res)
            }).catch((err)=>{
                message.error("There's been an error")
                setLoading(false)
            })
        }else{
            message.error("Please upload an image first")
        }
    }
    return (
        <>
            <Card
                title="Try Uploading An Image" 
                style={{width: '40%', minWidth: 400, border: '1px solid black', borderBottom:'0px'}} 
                headStyle={{display:'flex', justifyContent:'center'}}
                bodyStyle={{padding: "0px"}}
                cover={
                    <PhotoUpload/> 
                }
            >     
            </Card>
            <br/>
            {
                loading?(
                    <Spin/>
                ):(
                    <Button onClick={generateCaption} style={{border: '1px solid black'}}>Generate Caption!</Button>
                )
            }  
            <br/>
            <br/>
            {
                fetched?(
                    <>
                    <Card
                        style={{width: '40%', minWidth: 400, display:'flex', justifyContent:'center', border: '1px solid black'}} 
                    >
                        {fetched}
                    </Card>
                    <br/>
                    <br/>
                    </>
                    
                ):(
                    <></>
                )
            }
            
            
        </>
    )
}

export default ShowAndTell