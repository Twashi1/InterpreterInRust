use std::{error, string};

enum TokenType {
    NUMBER,
    OPERATOR
}

union TokenData {
    value: f32,
    operator: u8
}

struct Token {
    data_type: TokenType,
    data: TokenData
}

fn calc(operand1 : f32, operand2 : f32, operation : u8) -> f32 {
    match operation {
        b'+' => return operand1 + operand2,
        b'-' => return operand1 - operand2,
        b'/' => return operand1 / operand2,
        b'*' => return operand1 * operand2,
        _ => return 0.0
    }
}

fn readNumber(string : &String, pos : &mut usize) -> Token {
    let mut currentValue : f32 = 0.0;
    let mut digitsAfterDecimal : i32 = 0;
    
    while *pos < string.len() {
        let currentChar : u8 = string.as_bytes()[*pos];

        match currentChar {
            b'0'..=b'9' => {
                let digitValue = currentChar - b'0';
                
                if digitsAfterDecimal == 0 {
                    currentValue *= 10.0;
                    currentValue += digitValue as f32;
                } else {
                    currentValue += digitValue as f32 * (10 as f32).powf(-digitsAfterDecimal as f32) as f32;
                    digitsAfterDecimal += 1;
                }
            }
            b'.' => {
                if digitsAfterDecimal > 0 {
                    panic!("Multiple decimal points inside number");
                }

                digitsAfterDecimal = 1;
            }
            _ => return Token {data_type: TokenType::NUMBER, data: TokenData { value: currentValue }}
        }

        *pos += 1;
    }

    if currentValue != 0.0 {
        return Token { data_type: TokenType::NUMBER, data: TokenData{value : currentValue} }
    }
    
    panic!("Didn't read integer, or reached EOF");
}

fn lex(string : &String) -> Vec<Token> {
    let mut tokens : Vec<Token> = Vec::new();

    let mut currentPos : usize = 0;

    while currentPos < string.len() {
        let currentChar : u8 = string.as_bytes()[currentPos];

        match currentChar {
            b'0'..=b'9' | b'.' => {
                tokens.push(readNumber(string, &mut currentPos));
                currentPos -= 1;
            }
            b'+' | b'-' | b'*' | b'/' => {
                tokens.push(Token{ data_type: TokenType::OPERATOR, data: TokenData {operator: currentChar as u8} });
            }
            _ => {}
        }

        currentPos += 1;
    }

    return tokens;
}

fn parse_and_exec(token_stream: &Vec<Token>) -> f32 {
    let mut currentIndex : usize = 0;

    let mut currentResult : f32 = 0.0;

    let mut lhs: &Token = &token_stream[currentIndex];
    let mut op: &Token = &token_stream[currentIndex + 1];
    let mut rhs: &Token = &token_stream[currentIndex + 2];

    // Perform operation
    unsafe {
        currentResult = calc(lhs.data.value, rhs.data.value, op.data.operator);
    }

    currentIndex += 3;

    while currentIndex < token_stream.len() {
        let operator: &Token = &token_stream[currentIndex];
        let operand : &Token = &token_stream[currentIndex + 1];

        unsafe {
            currentResult = calc(currentResult, operand.data.value, operator.data.operator);
        }

        currentIndex += 2
    }
    
    return currentResult;
}

fn main() {
    let input : String = String::from("1+2*3");
    let tokens: Vec<Token> = lex(&input);
    let result : f32 = parse_and_exec(&tokens);

    println!("Res: {result}");
}
