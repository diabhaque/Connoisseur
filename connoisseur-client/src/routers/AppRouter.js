import React, { useContext } from 'react';
import { BrowserRouter as Router, Switch, Route, Redirect} from "react-router-dom";
import Layout from '../components/Layout'

import ShowAndTell from '../routes/ShowAndTell'
import ShowAndTellBts from '../routes/ShowAndTellBts'

import NotFoundPage from '../routes/NotFound'

import 'antd/dist/antd.css'


const AppRouter=()=>{
    return (
        <Router>
            <Layout>
                <Switch>
                    <Route path='/' component={ShowAndTell} exact={true}/>
                    <Route path='/show_and_tell_bts' component={ShowAndTellBts}/>
                    <Route component={NotFoundPage}/>
                </Switch> 
            </Layout>
        </Router>
    )
}

export default AppRouter