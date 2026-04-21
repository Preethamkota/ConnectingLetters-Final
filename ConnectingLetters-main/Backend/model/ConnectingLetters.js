import mongoose from "mongoose";
const ConnectingLettersSchema =  new mongoose.Schema({
    level: {
        type: Number,
        required: true
    },
    item: {
        type: Number,
        required: true
    },
    content:{
        type:mongoose.Schema.Types.Mixed,
        required: true
    }
    
});

const ConnectingLetter = mongoose.model('ConnectingLetter', ConnectingLettersSchema);
export default ConnectingLetter;