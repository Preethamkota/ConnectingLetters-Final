import express from 'express';
import mongoose from 'mongoose';
import ConnectingLetter from './model/ConnectingLetters.js';
import cors from 'cors';
const app = express();
const PORT = 5000;
app.use(express.json());
app.use(cors());

mongoose.connect('mongodb://localhost:27017/ConnectingLetters' , { 
    useNewUrlParser: true, 
    useUnifiedTopology: true 
});


app.get('/:level/:item', async (req, res) => {
    const { level, item } = req.params;
    console.log(level, item);
    try {
        const data =  await ConnectingLetter.findOne({level, item});
        res.json(data.content);
    } catch (error) {
        res.status(404).send('Data not found');
    }
    
});
app.post('/', async (req, res) => {
    const { level, item, content } = req.body;
    const newContent = new ConnectingLetter({level, item, content});
    try {
        await newContent.save();
        res.send('Data inserted successfully');
    } catch (error) {
        res.status(404).send('Data not inserted');
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});