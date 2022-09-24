import {
    getAllMarkets,
    getUserById,
    getFullMarket,
    placeBet
} from './api.js';

import {
    createWriteStream,
    createReadStream
} from 'fs';

import fetch from 'node-fetch'

import {
    writeFile
    //createWriteStream
} from 'fs/promises';
import { resolveObjectURL } from 'buffer';

let mode = "bet";

const TRADES_TO_COLLECT = 50;
let markets = await getAllMarkets();
let cachedUsers = [];
let examplesCollected = 0;

function sanitizeFilename(name) {
    return name
        .replace(/\s/g, "_")
        .replace("%", "")
        .replace("?", "")
        .replace(/\,/g, "")
        .replace(/\"/g, "")
        .replace(/\\/g, "-")
        .replace(/\//g, "-");
}

function stripOutput(name) {
    return name
        .replace("[", "")
        .replace("]", "");
}

const stream = createWriteStream("/temp/mdata.csv", { flag: "a" });

let labels =
    "Market ID, "
    + "Creator ID, "
    + "Market Creation Time, "
    + "Question, "
    + "Mechanism, "
    //+ "Resolution, "
    //    +" mkt.tags, "
    + "User ID, "
    + "User Creation Time, "
    + "User Monthly Profits, "
    + "User Profits, "
    + "Time of Bet Placement, "
    + "Prediction, "
    + "Amount, "
    + "Shares Purchased, "
    + "Prob Before, "
    + "Prob After, "
    + "Fees Paid, "
    + "Limit Prob, "
    + "Limit Size, "
    + "Limit Amount Paid, "
    + "Limit Shares Purchased, "
    + "Limit Order Cancelled"
    + "\n"
    ;

stream.write(labels);


for (let i = 0; i < markets.length && !(mode=="sample" && examplesCollected>=1); i++) {

    if (markets[i].outcomeType === "BINARY") {

        if (mode === "bet" && markets[i].isResolved == false) {
            try {
                let mkt = await getFullMarket(markets[i].id);
                if (mkt.bets.length == TRADES_TO_COLLECT) {
                    console.log(mkt.question);
                    let stream2 = createWriteStream("/temp/mdata.csv", { flag: "w" });
                    stream2.write(labels);
                    await mkt2csv(mkt, stream2);
                    stream2.end();
                    stream2.close();
                    stream2.destroy();

                    // let rr = createReadStream("/temp/mdata.csv");
                    // rr.on('readable', () => {
                    //     console.log(`readable: ${rr.read()}`);
                    // });
                    // rr.on('end', () => {
                    //     console.log('end');
                    // });

                    let prediction= stripOutput(await fetch("http://localhost:3000").then((res)=>{return res.text();}));

                    console.log(prediction+" vs. "+mkt.probability);

                    if (Math.abs(prediction-mkt.probability)>.05){
                    let bet = {
                        contractId: `${mkt.id}`,
                        outcome: null,
                        amount: 10
                    }
                    if (prediction>mkt.probability){bet.outcome="YES";}
                    else {bet.outcome="NO";}
                    console.log(bet);
                    //await placeBet(bet).then((resjson) => { console.log(resjson); });

                }


                }
            }
            catch (e) {
                console.log(e);
            }
            examplesCollected++;

        }
        else if ((mode === "collect" || mode === "sample") && markets[i].isResolved == true) {
            try {
                let mkt = await getFullMarket(markets[i].id);

                if ((mkt.resolution == "YES" || mkt.resolution != "NO") && mkt.bets.length >= TRADES_TO_COLLECT && mkt.bets.length < 1000) {

                    await mkt2csv(mkt, stream);
                    examplesCollected++;


                }
            } catch (e) {
                console.log(e);
                console.log("encountered an error");
            }
        }

    }
    //console.log("market " + i + " complete.");
}

async function mkt2csv(mkt, s) {

    let timeOfCollection = 0;
    if (mkt.bets.length == 0 ) {
        timeOfCollection = mkt.createdTime - 1;
    }
    else if (mkt.bets.length < TRADES_TO_COLLECT) {
        timeOfCollection = mkt.bets[mkt.bets.length - 1].createdTime - 1;
    }
    else {
        timeOfCollection = mkt.bets[TRADES_TO_COLLECT - 1].createdTime - 1;
    }

    for (let j = 0; j < TRADES_TO_COLLECT && j < mkt.bets.length; j++) {

        let thisBet = mkt.bets[mkt.bets.length - (j + 1)];
        let user = cachedUsers.find((u) => { thisBet.userId == u.id });
        if (user === undefined) {
            cachedUsers.push(await getUserById(thisBet.userId));
            user = cachedUsers.find((u) => { return thisBet.userId === u.id; });
        }

        let row = {
            mid: mkt.id,
            cid: mkt.creatorId,
            mtime: mkt.createdTime,
            question: sanitizeFilename(mkt.question),
            mechanism: mkt.mechanism,
            resolution: "",
            //    + mkt.tags+", "
            user: thisBet.userId,
            uage: user.createdTime,
            uprofitmonthly: user.profitCached.monthly,
            uprofit: user.profitCached.allTime,
            btime: thisBet.createdTime,
            bdirection: thisBet.outcome,
            bamount: thisBet.amount,
            bshares: thisBet.shares,
            pbefore: thisBet.probBefore,
            pafter: thisBet.probAfter,
            fees: 0,
            limit: thisBet.limitProb,
            lordersize: thisBet.orderAmount,
            lamount: 0,
            lshares: 0,
            lcancelled: thisBet.isCancelled

        }

        if (thisBet.fees === undefined) {
            row.fees = 0;
        }
        else {
            try {
                row.fees = (thisBet.fees.liquidityFee + thisBet.fees.platformFee + thisBet.fees.creatorFee);
            }
            catch (e) {
                console.log(e);
                row.fees = 0;
            }
        }

        if (mkt.resolution === "MKT") { row.resolution = mkt.resolutionProbability; }
        else if (mkt.resolution === "YES") { row.resolution = 1; }
        else if (mkt.resolution === "NO") { row.resolution = 0; }


        if (thisBet.limitProb !== undefined) {
            for (let k in thisBet.fills) {
                if (thisBet.fills[k].matchedBetId !== null) {
                    row.bshares -= thisBet.fills[k].shares;
                    row.bamount -= thisBet.fills[k].amount;
                    if (thisBet.fills[k].timestamp < timeOfCollection) {
                        row.lshares += thisBet.fills[k].shares;
                        row.lamount += thisBet.fills[k].amount;
                    }
                }
            }
        }

    

    
        stream.write(row.mid + ", "
    + row.cid + ", "
    + row.mtime + ", "
    + row.question + ", "
    + row.mechanism + ", "
    //+ row.resolution + ", "
    //    + mkt.tags+", "
    + row.user + ", "
    + row.uage + ", "
    + row.uprofitmonthly + ", "
    + row.uprofit + ", "
    + row.btime + ", "
    + row.bdirection + ", "
    + row.bamount + ", "
    + row.bshares + ", "
    + row.pbefore + ", "
    + row.pafter + ", "
    + row.fees + ", "
    + row.limit + ", "
    + row.lordersize + ", "
    + row.lamount + ", "
    + row.lshares + ", "
    + row.lcancelled
    + "\n"
        );

    }
}