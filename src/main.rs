use std::{error, string};

enum Operator {
    ADD,
    SUB,
    DIV,
    MUL
}

enum State {
    INTEGER_BEGIN,
    INTEGER,
    OPERATION
}

fn calc(operand1 : i32, operand2 : i32, operation : Operator) -> i32 {
    match operation {
        Operator::ADD => return operand1 + operand2,
        Operator::SUB => return operand1 - operand2,
        Operator::DIV => return operand1 / operand2,
        Operator::MUL => return operand1 * operand2
    }
}

fn exec(value : String) -> i32 {
    if value.len() == 0 { return 0; }

    let mut state : State = State::INTEGER_BEGIN;
    let mut currentValue : i32 = 0;
    let mut isNegative : bool = false;
    let mut operation : Operator = Operator::ADD;
    let mut store : Vec<i32> = Vec::new();

    let integerConvertClosure = | currentChar : u8, currentValue : &mut i32 | -> bool
    {
        // Check character is numeric
        if currentChar >= '0' as u8 && currentChar <= '9' as u8 {
            // Add to currentValue
            *currentValue = *currentValue * 10;
            *currentValue = *currentValue + (currentChar - '0' as u8) as i32;

            return true;
        }

        return false;
    };

    let errorClosure = |i : usize, msg : &str| -> () { println!("Parsing error idx {i}: {msg}"); };

    let mut i : usize = 0;

    while i < value.len() {
        let currentChar : u8 = value.as_bytes()[i];

        // No fallthrough, sad
        match state {
            State::INTEGER_BEGIN => {
                isNegative = currentChar == '-' as u8;
                
                integerConvertClosure(currentChar, &mut currentValue);

                // Transition to integer
                state = State::INTEGER;
            }
            State::INTEGER => {
                let wasDigit = integerConvertClosure(currentChar, &mut currentValue);

                // Assume transition to operation
                if !wasDigit {
                    store.push(currentValue * (!isNegative as i32 * 2 - 1));
                    currentValue = 0;
                    i -= 1;

                    state = State::OPERATION;
                }
            },
            State::OPERATION => {
                match currentChar as char {
                    '+' => { state = State::INTEGER_BEGIN; operation = Operator::ADD; },
                    '-' => { state = State::INTEGER_BEGIN; operation = Operator::SUB; },
                    '*' => { state = State::INTEGER_BEGIN; operation = Operator::MUL; },
                    '/' => { state = State::INTEGER_BEGIN; operation = Operator::DIV; },
                    ' ' => {},
                    _ => errorClosure(i, "Expected operation")
                }
            }
        }

        i += 1;
    }

    store.push(currentValue * (!isNegative as i32 * 2 - 1));

    if store.len() >= 2 {
        return calc(store[0], store[1], operation);
    }
    
    return 0;
}

fn main() {
    let result = exec("-35*-56".to_string());

    println!("{result}");
}
