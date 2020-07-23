import React from 'react'
import { Card } from 'antd'

const ShowAndTellBts =()=>{
    return (
        <>
            <Card
                title="Training the Model" 
                style={{width: '50%', minWidth: 500, minHeight: 200, border: '1px solid black',}} 
                headStyle={{display:'flex', justifyContent:'center', borderBottom: '1px solid black'}}
                bodyStyle={{padding: "0px"}}
            >     
            </Card>
            <br/>
            <br/>
            <Card
                title="Predictions" 
                style={{width: '50%', minWidth: 500, minHeight: 200, border: '1px solid black',}} 
                headStyle={{display:'flex', justifyContent:'center', borderBottom: '1px solid black'}}
                bodyStyle={{padding: "0px"}}
            >     
            </Card>
            <br/>
            <br/>
            <Card
                title="The Complete Picture" 
                style={{width: '50%', minWidth: 500, minHeight: 200, border: '1px solid black',}} 
                headStyle={{display:'flex', justifyContent:'center', borderBottom: '1px solid black'}}
                bodyStyle={{padding: "0px"}}
            >     
            </Card>
            <br/>
            <br/>
        </>
            
    )
}

export default ShowAndTellBts