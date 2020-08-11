import React, { useState, useContext } from 'react'
import { Upload, message } from 'antd'
import { LoadingOutlined, PlusOutlined } from '@ant-design/icons'
import ImageContext from '../context/image.context'
import ImgCrop from 'antd-img-crop'

const PhotoUpload=()=>{
    const [loading, setLoading] = useState(false)
    const [imageUrl, setImageUrl] = useState('')
    const [image, setImage] = useContext(ImageContext)
    const { Dragger } = Upload;

    const dummyRequest = ({ file, onSuccess }) => {
        setTimeout(() => {
          onSuccess("ok");
        }, 0);
    };

    const getBase64=(img, callback)=>{
        const reader = new FileReader();
        reader.addEventListener('load', () => callback(reader.result));
        reader.readAsDataURL(img);
    }

    const beforeUpload=(file)=>{
        const isJpgOrPng = file.type === 'image/jpeg';
        // || file.type === 'image/png';
        // Fix later
        if (!isJpgOrPng) {
          message.error('You can only upload JPG file!');
        }
        const isLt2M = file.size / 1024 / 1024 < 2;
        if (!isLt2M) {
          message.error('Image must smaller than 2MB!');
        }
        return isJpgOrPng && isLt2M;
    }

    const handleChange = info => {
        if (info.file.status === 'uploading') {
          setLoading(true);
          return;
        }
        if (info.file.status === 'done') {
            
            
            // if(info.fileList.length>1){
                
            // }
          // Get this url from response in real world.
          getBase64(info.file.originFileObj, (imageURL) =>{
            setLoading(false)
            setImageUrl(imageURL)
            setImage(imageURL)
            // console.log(imageURL)
          });
        }
    };

    const uploadButton =(
        <div style={{padding: '40px 0'}}>
            {loading? <LoadingOutlined/> : <PlusOutlined />}
            <div>Upload</div>
        </div>
    )

    return(
      <ImgCrop>
        <Dragger
              name="photoUpload"
              listType="picture-card"
              showUploadList={false}
              customRequest={dummyRequest}
              beforeUpload={beforeUpload}
              onChange={handleChange}
              style={{position: 'relative', left: '50%', transform: 'translateX(-50%)',backgroundColor: 'white', border: '1px solid black'}}

        >
              {Object.keys(image).length>1 ? <img src={image} alt="avatar" style={{ width: 'inherit', paddingTop: '20px'}} /> : uploadButton}
        </Dragger>
      </ImgCrop>    
        
    )
}

export default PhotoUpload