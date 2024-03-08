use core::panic;
use std::thread::current;

// https://rust-unofficial.github.io/patterns/patterns/behavioural/visitor.html

pub enum Node {
    None,
    Number(f32),
    BinaryAdd([Box<Node>; 2]),
    BinarySub([Box<Node>; 2]),
    BinaryMul([Box<Node>; 2]),
    BinaryDiv([Box<Node>; 2])
}

// Ideally, on any given node type, we could call
// visit(Node, &Node) -> Node
// This function performs a switch statement and calls the relevant method
fn visit(n: &Node) -> Node {
    match n {
        Node::BinaryAdd(children) => return visit_add(&children[0], &children[1]),
        Node::BinarySub(children) => return visit_sub(&children[0], &children[1]),
        Node::BinaryMul(children) => return visit_mul(&children[0], &children[1]),
        Node::BinaryDiv(children) => return visit_div(&children[0], &children[1]),
        Node::Number(value) => return Node::Number(*value),
        Node::None => return Node::None,
        _ => panic!("Couldn't extract value of node")
    }
}

fn visit_binary_number(left: &Node, right: &Node) -> [f32; 2] {
    let visited_left : Node = visit(left);
    let visited_right : Node = visit(right);

    // Attempt to convert both to number types
    if let Node::Number(left_value) = visited_left {
        if let Node::Number(right_value) = visited_right {
            return [left_value, right_value];
        }
    }

    panic!("One side of binary add was invalid");
}

fn visit_add(left: &Node, right: &Node) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right);

    return Node::Number(values[0] + values[1]);
}

fn visit_sub(left: &Node, right: &Node) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right);

    return Node::Number(values[0] - values[1]);
}

fn visit_div(left: &Node, right: &Node) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right);

    return Node::Number(values[0] / values[1]);
}

fn visit_mul(left: &Node, right: &Node) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right);

    return Node::Number(values[0] * values[1]);
}

enum Token {
    None,
    Number(f32),
    Operator(u8)
}

fn read_number(string : &String, pos : &mut usize) -> Token {
    let mut current_value : f32 = 0.0;
    let mut digits_after_decimal : i32 = 0;
    
    while *pos < string.len() {
        let current_char : u8 = string.as_bytes()[*pos];

        match current_char {
            b'0'..=b'9' => {
                let digit_value = current_char - b'0';
                
                if digits_after_decimal == 0 {
                    current_value *= 10.0;
                    current_value += digit_value as f32;
                } else {
                    current_value += digit_value as f32 * (10 as f32).powf(-digits_after_decimal as f32) as f32;
                    digits_after_decimal += 1;
                }
            }
            b'.' => {
                if digits_after_decimal > 0 {
                    panic!("Multiple decimal points inside number");
                }

                digits_after_decimal = 1;
            }
            _ => return Token::Number(current_value)
        }

        *pos += 1;
    }

    if current_value != 0.0 {
        return Token::Number(current_value);
    }
    
    panic!("Didn't read integer, or reached EOF");
}

fn lex(string : &String) -> Vec<Token> {
    let mut tokens : Vec<Token> = Vec::new();

    let mut current_pos : usize = 0;

    while current_pos < string.len() {
        let current_char : u8 = string.as_bytes()[current_pos];

        match current_char {
            b'0'..=b'9' | b'.' => {
                tokens.push(read_number(string, &mut current_pos));
                current_pos -= 1;
            }
            b'+' | b'-' | b'*' | b'/' => {
                tokens.push(Token::Operator(current_char));
            }
            _ => {}
        }

        current_pos += 1;
    }

    return tokens;
}

fn factor(token_index : &mut usize, token_stream : &Vec<Token>) -> Node {
    if let Token::Number(value) = token_stream[*token_index] {
        *token_index += 1;

        return Node::Number(value);
    }

    panic!("Expected number");
}

fn term(token_index : &mut usize, token_stream : &Vec<Token>) -> Node {
    let mut lhs: Node = factor(token_index, token_stream);

    if *token_index >= token_stream.len() {
        return lhs;
    }

    let mut current_token : &Token = &token_stream[*token_index];
    
    while *token_index < token_stream.len() {
        current_token = &token_stream[*token_index];

        if let Token::Operator(operator_type) = current_token {
            match operator_type {
                b'*' => {
                    *token_index += 1;
                    lhs = Node::BinaryMul([Box::new(lhs), Box::new(factor(token_index, token_stream))]);
                },
                b'/' => {
                    *token_index += 1;
                    lhs = Node::BinaryDiv([Box::new(lhs), Box::new(factor(token_index, token_stream))]);
                },
                _ => return lhs
            }
        } else {
            return lhs;
        }
    }

    return lhs;
}

fn expression(token_index : &mut usize, token_stream: &Vec<Token>) -> Node {
    let mut lhs: Node = term(token_index, token_stream);

    if *token_index >= token_stream.len() {
        return lhs;
    }

    let mut current_token : &Token = &token_stream[*token_index];
    
    while *token_index < token_stream.len() {
        current_token = &token_stream[*token_index];

        if let Token::Operator(operator_type) = current_token {
            match operator_type {
                b'+' => {
                    *token_index += 1;
                    lhs = Node::BinaryAdd([Box::new(lhs), Box::new(term(token_index, token_stream))]);
                },
                b'-' => {
                    *token_index += 1;
                    lhs = Node::BinarySub([Box::new(lhs), Box::new(term(token_index, token_stream))]);
                },
                _ => return lhs
            }
        } else {
            return lhs;
        }
    }

    return lhs;
}

// Build an AST from a token stream
// Recursive descent parser
fn parse(token_index : &mut usize, token_stream: &Vec<Token>) -> Node {
    while *token_index < token_stream.len() {
        let current_token : &Token = &token_stream[*token_index];

        // TODO: badly formed right now
        match current_token {
            Token::Number(_) => return expression(token_index, token_stream),
            _ => return Node::None
        }

        // *token_index += 1;
    }
    
    return Node::None;
}

fn interpret(tree: &Node) -> Node {
    return visit(tree);
}

fn main() {
    let input : String = String::from("3*6 + 4 - 7 / 3");
    let tokens: Vec<Token> = lex(&input);
    let mut token_index : usize = 0;
    let tree: Node = parse(&mut token_index, &tokens);
    let result : Node = interpret(&tree);

    if let Node::Number(value) = result {
        println!("Res: {value}");
    } else {
        panic!("Interpreter didn't return number");
    }
}
